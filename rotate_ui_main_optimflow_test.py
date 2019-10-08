from PyQt5 import QtCore,QtWidgets
# from PyQt5.QtGui import QMouseEvent

from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox,QInputDialog,QLineEdit,QMainWindow
from PyQt5.QtCore import Qt,QEvent

from os import execl,listdir,environ
from pydicom import dcmread
from scipy.io import loadmat,savemat
import cv2

import numpy as np
from numpy import pi,sin,cos,sqrt
from numpy.linalg import inv
from skimage.transform import resize
from skimage.morphology import skeletonize_3d
# from h5py import File

from tkinter.filedialog import askdirectory,askopenfilename,asksaveasfilename
from tkinter import Tk

import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit


import rotate_ui_optimflow,sys

# from RotateGPU import rotate_volume


class RotateWindow(QMainWindow,rotate_ui_optimflow.Ui_MainWindow):
	def __init__(self,parent=None):

		super(RotateWindow, self).__init__(parent)
		self.setupUi(self)
		# super().setupUi(MainWindow)
		x=0
		y=0
		self.CAG_OG.installEventFilter(self)
		self.text = "x: {0},  y: {1}".format(x, y)

		self.LOAD_CT.clicked.connect(self.load_ct)
		self.LOAD_CAG.clicked.connect(self.load_cag)
		self.LOAD_CAG_MAT.clicked.connect(self.load_cag_mat)

		self.SAVE_CT.clicked.connect(self.save_CT)
		self.CT_VOLUME_SAVE.clicked.connect(self.save_CT_volume)
		self.SAVE_CT_SK_MASK.clicked.connect(self.save_CT_sk_mask)
		self.SAVE_CAG.clicked.connect(self.save_CAG)


		self.CAG_BAR.valueChanged.connect(self.CAG_BAR_ch)
		self.CAG_frame.returnPressed.connect(self.CAG_frame_input)

		self.OP_FLOW.stateChanged.connect(self.optime_flow_sw)

		self.RAO_LAO.valueChanged.connect(self.RAOLAO_value_ch)
		self.RAO_LAO.sliderReleased.connect(self.RAOLAO_value_re)

		self.CRA_CAU.valueChanged.connect(self.CRACAU_value_ch)
		self.CRA_CAU.sliderReleased.connect(self.CRACAU_value_re)

		self.CRA_CAU_input.returnPressed.connect(self.Rotate_textinput)
		self.RAO_LAO_input.returnPressed.connect(self.Rotate_textinput)

		self.RESET.clicked.connect(self.restart_program)

		self.point_array=np.load('D:/CTA2CGA/heart_py_code/Rotate_project_GUI/CVAI0398_60_1_centerline_pointarray_reshape.npy').astype(np.float32)
		self.point_array_og=np.load('D:/CVAI0398_60_1_centerlinr_point.npy')


		self._rotation_kernel_source = """
		texture<float, cudaTextureType3D, cudaReadModeElementType> tex;

		__global__ void copy_texture_kernel(
		    const float ox,const float oy,const float oz,
		    const float r11,const float r12,const float r13,
		    const float r21,const float r22,const float r23,
		    const float r31,const float r32,const float r33, 
		    unsigned short oldiz, unsigned short oldiw, unsigned short oldih,
		    unsigned short newiz, unsigned short newiw, unsigned short newih,
		    float* data) {

		        // calculate index (i,j,k)
		        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		        unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
		        
		        // out side the Range
		        if((x >= newiw) || (y >= newih) || (z >= newiz))
		            return;
		        
		        // flaten index
		        unsigned int didx = x + y*newih + z*newiw*newih;

		        float sx = ( x*r11 + y*r12 + z*r13 ) + ox;
		        float sy = ( x*r21 + y*r22 + z*r23 ) + oy;
		        float sz = ( x*r31 + y*r32 + z*r33 ) + oz;
		        
		        if( (sx < 0) || (sx >= oldiw) || (sy < 0) || (sy >= oldih) || (sz < 0) || (sz >= oldiz) ) { 
		            data[didx] = 0; 
		            return;
		        }

		        data[didx] = tex3D(tex, sz, sy, sx);
		    }
		"""
		self.mod_3d_rotation=pycuda.compiler.SourceModule(self._rotation_kernel_source )
		self.copy_texture_func = self.mod_3d_rotation.get_function("copy_texture_kernel")
		self.texref=self.mod_3d_rotation.get_texref('tex')

	def eventFilter(self, source, event):
		if event.type() == QEvent.MouseMove and source is self.CAG_OG:
			x = event.x()
			y = event.y()
			text = "x: {0},  y: {1}".format(x, y)
			self.STATE.setText(text)
		return QtWidgets.QMainWindow.eventFilter(self, source, event)

	def rotate_volume(self,a, angle = [0,0], blocks = (16,16,2) ):
		CAU = angle[1]/180*pi
		LR = angle[0]/180*pi

		Ry=np.array(([cos(CAU),0,sin(CAU)],[0,1,0],[-sin(CAU),0,cos(CAU)]),dtype=np.float32)
		Rz=np.array(([cos(LR),-sin(LR),0],[sin(LR),cos(LR),0],[0,0,1]),dtype=np.float32)

		rotate_matrix=Ry.dot(Rz)
		inv_rot_matrix=inv(rotate_matrix)

		a = a.astype("float32")
		z,n,m=a.shape

		# new dimensions
		size_rot=int(sqrt(m**2+n**2+z**2))
		new_image_dim = [size_rot,size_rot,size_rot]
		output = np.zeros(new_image_dim,dtype=np.float32)

		m_ro,n_ro,z_ro=output.shape

		m2=0.5*(m)
		n2=0.5*(n)
		z2=0.5*(z)
		nm2=0.5*(m_ro)
		nn2=0.5*(n_ro)
		nz2=0.5*(z_ro)

		ox=nm2*-inv_rot_matrix[0,0]+(nn2*-inv_rot_matrix[0,1])+(nz2*-inv_rot_matrix[0,2])
		oy=nm2*-inv_rot_matrix[1,0]+(nn2*-inv_rot_matrix[1,1])+(nz2*-inv_rot_matrix[1,2])
		oz=nm2*-inv_rot_matrix[2,0]+(nn2*-inv_rot_matrix[2,1])+(nz2*-inv_rot_matrix[2,2])
		ox,oy,oz=ox+m2,oy+n2,oz+z2

		a_cuda=self.numpy3d_to_array(a)
		self.texref.set_array(a_cuda)
		# texref.set_filter_mode(cuda.filter_mode.LINEAR)

		# gridsize
		gridx = new_image_dim[0]/blocks[0] if \
				new_image_dim[0]%blocks[0]==1 else new_image_dim[0]/blocks[0] +1
		gridy = new_image_dim[1]/blocks[1] if \
			new_image_dim[1]%blocks[1]==1 else new_image_dim[1]/blocks[1] +1
		gridz = new_image_dim[2]/blocks[2] if \
				new_image_dim[2]%blocks[2]==1 else new_image_dim[2]/blocks[2] +1
	    
	    # Call the kernel
		self.copy_texture_func(
			np.float32(ox),np.float32(oy),np.float32(oz),
			np.float32(inv_rot_matrix[0,0]),np.float32(inv_rot_matrix[0,1]),np.float32(inv_rot_matrix[0,2]),
			np.float32(inv_rot_matrix[1,0]),np.float32(inv_rot_matrix[1,1]),np.float32(inv_rot_matrix[1,2]),
			np.float32(inv_rot_matrix[2,0]),np.float32(inv_rot_matrix[2,1]),np.float32(inv_rot_matrix[2,2]),
			np.uint16(z),np.uint16(n), np.uint16(m),
			np.uint16(new_image_dim[2]),np.uint16(new_image_dim[1]), np.uint16(new_image_dim[0]),
			cuda.Out(output),
			texrefs=[self.texref],
			block=blocks,
			grid=(int(gridx),int(gridy),int(gridz)))

		return output

	def numpy3d_to_array(self,np_array, allow_surface_bind=True):

		d, h, w = np_array.shape

		descr = cuda.ArrayDescriptor3D()
		descr.width = w
		descr.height = h
		descr.depth = d
		descr.format = cuda.dtype_to_array_format(np_array.dtype)
		descr.num_channels = 1
		descr.flags = 0

		if allow_surface_bind:
			descr.flags = cuda.array3d_flags.SURFACE_LDST

		device_array = cuda.Array(descr)

		copy = cuda.Memcpy3D()
		copy.set_src_host(np_array)
		copy.set_dst_array(device_array)
		copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
		copy.src_height = copy.height = h
		copy.depth = d

		copy()

		return device_array

	def getText(self,matdata):
		inputBox=QInputDialog()
		inputBox.setInputMode(0)
		inputBox.setWindowTitle('MatFileKeyInputDialog')
		itemlist=list()
		for key in matdata.keys():
			itemlist.append(key)
		inputBox.setComboBoxItems(itemlist)
		inputBox.setComboBoxEditable(False)
		inputBox.setLabelText('Please Input MatFile Key')
		inputBox.setOkButtonText(u'Ok')
		inputBox.setCancelButtonText(u'Cancel')
		if inputBox.exec_() and inputBox.textValue()!='':
			return inputBox.textValue()

	def CTProjection(self,CTSet,Ptype='meanIP'):
		m,n,z=CTSet.shape
		project=np.zeros((m,n,3))

		if Ptype=='meanIP':
			project[:,:,0]=CTSet.sum(2)/z
			project[:,:,1]=CTSet.sum(2)/z
			project[:,:,2]=CTSet.sum(2)/z
			return project
		if Ptype=='maxIP':
			project[:,:,0]=CTSet.max(2)
			project[:,:,1]=CTSet.max(2)
			project[:,:,2]=CTSet.max(2)
			return project

	def NormlizDcm(self,dicom_set):

		dcm_float=dicom_set.astype(np.float)
		dcm_uint8=np.zeros(dicom_set.shape,dtype=np.float32)

		dcm_float[dcm_float>600]=600
		dcm_float[dcm_float<-200]=-200
		dcm_uint8=255*((dcm_float-dcm_float.min())/(dcm_float.max()-dcm_float.min()))
		return dcm_uint8

	def norlize_255(self,image):
		# print('normalizing')
		top=image.max()
		bot=image.min()

		output=255*((image-bot)/(top-bot))
		output=output.astype(np.float32)

		return output

	def load_ct(self):
		global TDM_wv,TDM_wv_resize,vessel
		root=Tk()
		root.withdraw()
		fille_path=askopenfilename(filetypes = (("mat files","*.mat"),("all files","*.*")))
		try:
			TDM_data=loadmat(fille_path)
		except NameError:
			pass
		else:
			TDM_data=loadmat(fille_path)

			TDM_key=self.getText(TDM_data)
			TDM=TDM_data[TDM_key].astype(np.float32)

			vessel_key=self.getText(TDM_data)
			vessel=TDM_data[vessel_key].astype(np.float32)

			TDM=self.NormlizDcm(TDM)
			vessel=self.norlize_255(vessel)
			
			TDM_wv=(18*vessel)+TDM
			m,n,z=TDM_wv.shape

			TDM_wv_resize=resize(TDM_wv,(int(m/3),int(n/3),int(z/3)))
			tpTDM_wv=np.transpose(TDM_wv)
			print('TDM_wv_resize size = ',TDM_wv_resize.shape)

			# TDM_wv_resize.resize(int(m/2),int(n/2),int(z/2))

			ct_proj=self.CTProjection(tpTDM_wv)
			# ct_proj=self.norlize_255(ct_proj)
			ct_proj=255-ct_proj.astype(np.uint8)
			print('ct_proj shape = ',ct_proj.shape)
			height,width,bytesPerComponent=ct_proj.shape
			bytesPerLine=3*width
			QImg=QImage(ct_proj.data, width, height, bytesPerLine, QImage.Format_RGB888)
			pixmap=QPixmap.fromImage(QImg)
			self.CT_SIM.setPixmap(pixmap)
			

	def optime_flow_sw(self,state):
		global frame_RGB,old_gray,mask,p0,lk_params,color
		try:
			CAG_array
		except NameError:
			pass
		else:
			if state==Qt.Checked:
				lk_params = dict(winSize=(100, 100),maxLevel=30,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0.04))
				color = np.random.randint(0, 255, (100, 3))
				init_frame_num=self.CAG_BAR.value()
				old_gray=cv2.medianBlur(CAG_array[init_frame_num,:,:],5)
				init_gray=CAG_array[init_frame_num,:,:].copy()
				p0=self.point_array
				mask = np.zeros((old_gray.shape[0],old_gray.shape[1],3),dtype=np.uint8)
				frame_RGB = np.zeros((old_gray.shape[0],old_gray.shape[1],3),dtype=np.uint8)
				for i in range(len(self.point_array_og)):
					a=self.point_array_og[i,0]
					b=self.point_array_og[i,1]
					init_gray[a,b]=255

				height,width=CAG_array[init_frame_num,:,:].shape
				bytesPerLine=1*width
				QImg=QImage(init_gray.data, width, height, bytesPerLine, QImage.Format_Indexed8)
				pixmap=QPixmap.fromImage(QImg)
				self.CAG_OG.setPixmap(pixmap)
				

				p0=self.point_array
				mask = np.zeros((old_gray.shape[0],old_gray.shape[1],3),dtype=np.uint8)
				frame_RGB = np.zeros((old_gray.shape[0],old_gray.shape[1],3),dtype=np.uint8)


	def load_cag(self):
		global CAG_array
		root=Tk()
		root.withdraw()
		fille_path=askopenfilename()
		try:
			CAG_data=dcmread(fille_path)
		except NameError:
			pass
		else:
			CAG=dcmread(fille_path)
			CAG_array=CAG.pixel_array

			z,m,n=CAG_array.shape

			CAG_RAOLAO=CAG.PositionerPrimaryAngle
			CAG_CRACAU=CAG.PositionerSecondaryAngle

			self.CAG_BAR.setMinimum(0)
			self.CAG_BAR.setMaximum(z-1)

			if CAG_RAOLAO>0:
				RAO_LAO_txt='LAO : ' +str(CAG_RAOLAO)
			else:
				RAO_LAO_txt='RAO : ' +str(CAG_RAOLAO)
				# rotate_deg='RAO/LAO :  '+str(CAG_RAOLAO)+'  CRA/CAU :  '+str(CAG_CRACAU)

			if CAG_CRACAU>0:
				CRA_CAU_txt='CRA : ' +str(CAG_CRACAU)
			else:
				CRA_CAU_txt='CAU : ' +str(CAG_CRACAU)

			rotate_deg=RAO_LAO_txt+' , '+CRA_CAU_txt

			self.CAG_DEG.setText(rotate_deg)

			height,width=CAG_array[0,:,:].shape
			bytesPerLine=1*width
			QImg=QImage(CAG_array[0,:,:].data, width, height, bytesPerLine, QImage.Format_Indexed8)
			pixmap=QPixmap.fromImage(QImg)
			self.CAG_OG.setPixmap(pixmap)
			

	def load_cag_mat(self):
		global CAG_array
		root=Tk()
		root.withdraw()
		fille_path=askopenfilename()
		try:
			CAG_mat=loadmat(fille_path)
		except NameError:
			pass
		else:
			CAG_mat=loadmat(fille_path)
			CAG_key=self.getText(CAG_mat)
			CAG_array_mat=CAG_mat[CAG_key]
			CAG_array=CAG_array_mat.copy()

			z,m,n=CAG_array.shape
			# print(m,n,z)

			self.CAG_BAR.setMinimum(0)
			self.CAG_BAR.setMaximum(z-1)

			height,width=CAG_array[0,:,:].shape
			print(height,width)
			bytesPerLine=1*width
			QImg=QImage(CAG_array[0,:,:].data.tobytes(), width, height, bytesPerLine, QImage.Format_Indexed8)
			pixmap=QPixmap.fromImage(QImg)
			self.CAG_OG.setPixmap(pixmap)
			

	def CAG_BAR_ch(self):
		try:
			CAG_array
		except NameError:
			pass
		else:
			optimFlow_tag=self.OP_FLOW.isChecked()
			# self.STATE.setText(str(a))
			if optimFlow_tag == False:
				frame_num=self.CAG_BAR.value()

				height,width=CAG_array[frame_num,:,:].shape
				bytesPerLine=1*width
				QImg=QImage(CAG_array[frame_num,:,:].data, width, height, bytesPerLine, QImage.Format_Indexed8)
				pixmap=QPixmap.fromImage(QImg)
				self.CAG_OG.setPixmap(pixmap)
				

				self.CAG_frame.setText(str(frame_num))

			else:
				global frame_RGB,old_gray,mask,p0,lk_params,color

				frame_num=self.CAG_BAR.value()
				frame_gray = cv2.medianBlur(CAG_array[frame_num,:,:],5)

				frame_RGB[:,:,0]=frame_gray[:,:]
				frame_RGB[:,:,1]=frame_gray[:,:]
				frame_RGB[:,:,2]=frame_gray[:,:]


				p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
				good_new = p1[st == 1]
				good_old = p0[st == 1]
				for i, (new, old) in enumerate(zip(good_new, good_old)):
					a, b = new.ravel()
					# a, b = old.ravel() 
					a,b=int(a),int(b)
					if a>511:
						a=511
					if b>511:
						b=511
					# mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
					# frame_RGB = cv2.circle(frame_RGB, (a, b), 2, color[i].tolist(), -1)
					frame_RGB[a,b,0]=255
					frame_RGB[a,b,1]=255
					frame_RGB[a,b,2]=255
					# img = cv2.add(frame_RGB, mask)
				img = frame_RGB.copy()

				height,width=CAG_array[frame_num,:,:].shape

				bytesPerLine=3*width
				QImg=QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
				pixmap=QPixmap.fromImage(QImg)
				self.CAG_OG.setPixmap(pixmap)
				

				old_gray = frame_gray.copy()
				p0 = good_new.reshape(-1, 1, 2)
				self.CAG_frame.setText(str(frame_num))

	def CAG_frame_input(self):
		try:
			CAG_array
		except NameError:
			pass
		else:
			frame_num=int(self.CAG_frame.text())

			height,width=CAG_array[frame_num,:,:].shape
			bytesPerLine=1*width
			QImg=QImage(CAG_array[frame_num,:,:].data, width, height, bytesPerLine, QImage.Format_Indexed8)
			pixmap=QPixmap.fromImage(QImg)
			self.CAG_OG.setPixmap(pixmap)
			
			self.CAG_BAR.setValue(frame_num)

	def RAOLAO_value_ch(self):
		try:
			TDM_wv_resize
		except NameError:
			pass
		else:
			RAOLAO_deg=self.RAO_LAO.value()
			CRACAU_deg=self.CRA_CAU.value()

			if RAOLAO_deg>0:
				RAO_LAO_txt='LAO : ' +str(RAOLAO_deg)
			else:
				RAO_LAO_txt='RAO : ' +str(RAOLAO_deg)

			if CRACAU_deg>0:
				CRA_CAU_txt='CRA : ' +str(CRACAU_deg)
			else:
				CRA_CAU_txt='CAU : ' +str(CRACAU_deg)

			rotate_deg=RAO_LAO_txt+' , '+CRA_CAU_txt

			# rotate_deg='RAO/LAO :  '+str(RAOLAO_deg)+'  CRA/CAU :  '+str(CRACAU_deg)
			self.ROTATE_DEG.setText(rotate_deg)

			deg=[RAOLAO_deg,CRACAU_deg]
			ct_rotate=self.rotate_volume(TDM_wv_resize.copy(),angle=deg)
			m,n,z=ct_rotate.shape
			w=512/6
			
			a1=int((m/2)-w)
			a2=int((m/2)+w)
			
			b1=int((n/2)-w)
			b2=int((n/2)+w)

			proj=self.CTProjection(ct_rotate[a1:a2,b1:b2,:])
			proj=self.norlize_255(proj)
			proj=255-proj.astype(np.uint8)

			self.RAO_LAO_input.setText(str(RAOLAO_deg))

			height,width,bytesPerComponent=proj.shape
			bytesPerLine=3*width
			QImg=QImage(proj.data, width, height, bytesPerLine, QImage.Format_RGB888)
			pixmap=QPixmap.fromImage(QImg)
			self.CT_SIM.setPixmap(pixmap)
			self.CT_SIM.setScaledContents(True)

	def RAOLAO_value_re(self):

		try:
			TDM_wv
		except NameError:
			pass
		else:
			global proj
			
			RAOLAO_deg=self.RAO_LAO.value()
			CRACAU_deg=self.CRA_CAU.value()
			deg=[RAOLAO_deg,CRACAU_deg]
			ct_rotate=self.rotate_volume(TDM_wv.copy(),angle=deg)

			m,n,z=ct_rotate.shape
			m2,n2=int(m/2),int(n/2)

			proj=self.CTProjection(ct_rotate[m2-256:m2+256,n2-256:n2+256,:])
			proj=self.norlize_255(proj)
			proj=255-proj.astype(np.uint8)

			height,width,bytesPerComponent=proj.shape
			bytesPerLine=3*width
			QImg=QImage(proj.data, width, height, bytesPerLine, QImage.Format_RGB888)
			pixmap=QPixmap.fromImage(QImg)
			self.CT_SIM.setPixmap(pixmap)
			self.CT_SIM.setScaledContents(True)
			

	def CRACAU_value_ch(self):
		try:
			TDM_wv_resize
		except NameError:
			pass
		else:

			RAOLAO_deg=self.RAO_LAO.value()
			CRACAU_deg=self.CRA_CAU.value()

			if RAOLAO_deg>0:
				RAO_LAO_txt='LAO : ' +str(RAOLAO_deg)
			else:
				RAO_LAO_txt='RAO : ' +str(RAOLAO_deg)

			if CRACAU_deg>0:
				CRA_CAU_txt='CRA : ' +str(CRACAU_deg)
			else:
				CRA_CAU_txt='CAU : ' +str(CRACAU_deg)

			rotate_deg=RAO_LAO_txt+' , '+CRA_CAU_txt

			self.ROTATE_DEG.setText(rotate_deg)

			deg=[RAOLAO_deg,CRACAU_deg]
			ct_rotate=self.rotate_volume(TDM_wv_resize.copy(),angle=deg)
			m,n,z=ct_rotate.shape
			w=512/6
			
			a1=int((m/2)-w)
			a2=int((m/2)+w)
			
			b1=int((n/2)-w)
			b2=int((n/2)+w)

			proj=self.CTProjection(ct_rotate[a1:a2,b1:b2,:])
			proj=self.norlize_255(proj)
			proj=255-proj.astype(np.uint8)
			self.CRA_CAU_input.setText(str(CRACAU_deg))

			height,width,bytesPerComponent=proj.shape
			bytesPerLine=3*width
			QImg=QImage(proj.data, width, height, bytesPerLine, QImage.Format_RGB888)
			pixmap=QPixmap.fromImage(QImg)
			self.CT_SIM.setPixmap(pixmap)
			self.CT_SIM.setScaledContents(True)

	def CRACAU_value_re(self):

		try:
			TDM_wv
		except NameError:
			pass
		else:
			global proj

			RAOLAO_deg=self.RAO_LAO.value()
			CRACAU_deg=self.CRA_CAU.value()
			deg=[RAOLAO_deg,CRACAU_deg]
			ct_rotate=self.rotate_volume(TDM_wv.copy(),angle=deg)

			m,n,z=ct_rotate.shape
			m2,n2=int(m/2),int(n/2)

			proj=self.CTProjection(ct_rotate[m2-256:m2+256,n2-256:n2+256,:])
			proj=self.norlize_255(proj)
			proj=255-proj.astype(np.uint8)

			height,width,bytesPerComponent=proj.shape
			bytesPerLine=3*width
			QImg=QImage(proj.data, width, height, bytesPerLine, QImage.Format_RGB888)
			pixmap=QPixmap.fromImage(QImg)
			self.CT_SIM.setPixmap(pixmap)
			self.CT_SIM.setScaledContents(False)

	def Rotate_textinput(self):
		try:
			TDM_wv
		except NameError:
			pass
		else:
			global proj

			RAOLAO_deg=self.RAO_LAO_input.text()
			CRACAU_deg=self.CRA_CAU_input.text()

			self.RAO_LAO.setValue(float(RAOLAO_deg))
			self.CRA_CAU.setValue(float(CRACAU_deg))

			if float(RAOLAO_deg)>0:
				RAO_LAO_txt='LAO : ' +RAOLAO_deg
			else:
				RAO_LAO_txt='RAO : ' +RAOLAO_deg

			if float(CRACAU_deg)>0:
				CRA_CAU_txt='CRA : ' +CRACAU_deg
			else:
				CRA_CAU_txt='CAU : ' +CRACAU_deg

			rotate_deg=RAO_LAO_txt+' , '+CRA_CAU_txt

			self.ROTATE_DEG.setText(rotate_deg)

			deg=[float(RAOLAO_deg),float(CRACAU_deg)]
			ct_rotate=self.rotate_volume(TDM_wv.copy(),angle=deg)

			m,n,z=ct_rotate.shape
			m2,n2=int(m/2),int(n/2)

			proj=self.CTProjection(ct_rotate[m2-256:m2+256,n2-256:n2+256,:])
			proj=self.norlize_255(proj)
			proj=255-proj.astype(np.uint8)

			height,width,bytesPerComponent=proj.shape
			bytesPerLine=3*width
			QImg=QImage(proj.data, width, height, bytesPerLine, QImage.Format_RGB888)
			pixmap=QPixmap.fromImage(QImg)
			self.CT_SIM.setPixmap(pixmap)
			

	def save_CT(self):
		try:
			proj
		except NameError:
			pass
		else:
			root=Tk()
			root.withdraw()
			save_path=asksaveasfilename(initialdir = "D:/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
			proj_save=proj.copy()
			# proj_save[proj!=255]=0
			# proj_save=255-proj_save
			if save_path=='':
				pass
			else:
				cv2.imwrite(save_path,proj_save)

	def save_CAG(self):
		try:
			CAG_array
		except NameError:
			pass
		else:
			frame_num=self.CAG_BAR.value()
			CAG_save_frame=CAG_array[frame_num,:,:]
			root=Tk()
			root.withdraw()
			save_path=asksaveasfilename(initialdir = "D:/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
			# print(save_path)

			if save_path=='':
				pass
			else:
				cv2.imwrite(save_path,CAG_save_frame)

	def save_CT_volume(self):
		try:
			TDM_wv
		except NameError:
			pass
		else:
			self.STATE.setText('saving data ....')
			root=Tk()
			root.withdraw()
			save_path=asksaveasfilename(initialdir = "D:/",title = "Select file",filetypes = (("mat files","*.mat"),("all files","*.*")))

			RAOLAO_deg=self.RAO_LAO.value()
			CRACAU_deg=self.CRA_CAU.value()
			deg=[RAOLAO_deg,CRACAU_deg]
			ct_rotate=self.rotate_volume(TDM_wv.copy(),angle=deg)

			rotate_tag='RAOLAO_deg'+str(RAOLAO_deg)+'CRACAU_deg'+str(CRACAU_deg)
			savemat(save_path,{rotate_tag:ct_rotate})

			self.STATE.setText('done')

	def save_CT_sk_mask(self):
		try:
			vessel
		except NameError:
			pass
		else:
			self.STATE.setText('saving data ....')

			root=Tk()
			root.withdraw()
			save_path=asksaveasfilename(initialdir = "D:/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))

			RAOLAO_deg=self.RAO_LAO.value()
			CRACAU_deg=self.CRA_CAU.value()
			deg=[RAOLAO_deg,CRACAU_deg]
			ct_rotate=self.rotate_volume(vessel.copy(),angle=deg)
			ct_rotate[ct_rotate>0]=1

			m,n,z=ct_rotate.shape
			m2,n2=int(m/2),int(n/2)

			save_data=skeletonize_3d(ct_rotate)
			proj_sk=self.CTProjection(save_data[m2-256:m2+256,n2-256:n2+256,:])
			proj_sk[proj_sk>0]=255

			# proj_sk=self.norlize_255(proj_sk)
			if save_path=='':
				pass
			else:
				cv2.imwrite(save_path,proj_sk)
			self.STATE.setText('done')


	def restart_program(self):
		python = sys.executable
		execl(python, python, * sys.argv)

if __name__=='__main__':
	app=QtWidgets.QApplication(sys.argv)
	# MainWindow=QtWidgets.QMainWindow()
	MainWindow=RotateWindow()
	# ui=RotateWindow(MainWindow)  
	MainWindow.setWindowTitle('CAG rotate tool')
	# app.installEventFilter(MainWindow)

	MainWindow.show()
	sys.exit(app.exec_())