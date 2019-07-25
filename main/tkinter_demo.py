import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import tkinter as tk
import numpy as np
import cv2,os,re
from PIL import Image, ImageTk
from facenet.recognize import Facenet_Recognize
window = tk.Tk()
window.title('Demo')
window.geometry('600x800')


# MTCNN 模型
test_mode = "onet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)


# facenet 模型
model_dir = '../facenet/models/20170512-110547.pb'
image_size = 160
npz_file = '../facenet/data.npz'
face_recognize = Facenet_Recognize(model_dir,image_size,npz_file)

# 获取摄像头
video_capture = cv2.VideoCapture(sys.argv[1])
video_capture.set(3, 300)
video_capture.set(4, 400)
# fps = video_capture.get(cv2.CAP_PROP_FPS)


# 变量
name = ''
# most_similiar_face = ''
start_face_detect = False
start_face_recog = False
btn_face_detect = tk.StringVar()
btn_face_recog = tk.StringVar()
btn_face_detect.set('开始检测人脸')
btn_face_recog.set('开始识别人脸')


frame_W = 600
frame_H = 450
detect_W = 400
detect_H = 300

def vedio_loop():
	global frame
	global boxes_c
	global name
	ret,frame = video_capture.read()
	if ret is True:
		frame = cv2.resize(frame,(frame_W,frame_H),interpolation=cv2.INTER_CUBIC)

		if start_face_detect is True:
			image = np.array(frame)
			image = cv2.resize(image,(detect_W,detect_H),interpolation=cv2.INTER_CUBIC)
			boxes_c, _ = mtcnn_detector.detect(image)
			boxes_c *= frame_W/detect_W

			for i in range(boxes_c.shape[0]):
				bbox = boxes_c[i, :4]
				corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
				cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),(corpbbox[2], corpbbox[3]), (0,155,255), 2)

			# 显示最大框名字
			if start_face_recog is True and len(boxes_c) > 0:
				pic,corpbbox = crop_max_face(frame,boxes_c)
				path = face_recognize.predict(pic)[0][0]
				ava = cv2.imread('../face_data/'+path)
				ava = cv2.resize(ava,(100,100),interpolation=cv2.INTER_CUBIC)
				name = path.split('.')[0].split('_')[:-1]
				cv2.putText(frame, '_'.join(name), (corpbbox[0], corpbbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
				frame[:100,:100] = ava


		cov= cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
		img=Image.fromarray(cov)
		img=ImageTk.PhotoImage(img)
		canvas.create_image(300,0,anchor=tk.N,image=img) 
		window.update_idletasks()
		window.after(1,vedio_loop)


def crop_max_face(frame,boxes_c):
	max_index = np.argmax((boxes_c[:,2]-boxes_c[:,0])*(boxes_c[:,3]-boxes_c[:,1]))
	bbox = boxes_c[max_index,:4]
	corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
	pic = frame[corpbbox[1]:corpbbox[3],corpbbox[0]:corpbbox[2]]
	pic = cv2.resize(pic,(image_size,image_size),interpolation=cv2.INTER_CUBIC)
	return pic,corpbbox

def crop_face(frame,boxes_c):
	res = []
	for box in boxes_c:
		bbox = box[:4]
		corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
		pic = frame[corpbbox[1]:corpbbox[3],corpbbox[0]:corpbbox[2]]
		pic = cv2.resize(pic,(image_size,image_size),interpolation=cv2.INTER_CUBIC)
		res.append(pic)
	return np.array(res)

# 保存人脸
def save_face():
	global frame
	global boxes_c
	if len(boxes_c) > 0:
		pic,_ = crop_max_face(frame,boxes_c)
		cv2.imwrite('../face_data/unname.jpg',pic)

# 按钮人脸检测
def face_detection():
	global start_face_detect
	global start_face_recog
	if start_face_detect is True:
		start_face_detect = False
		start_face_recog = False
		btn_face_detect.set('开始人脸检测')
		btn_face_recog.set('开始识别人脸')
	else:
		start_face_detect = True
		btn_face_detect.set('停止人脸检测')

# 按钮人脸识别
def face_recognition():
	global start_face_recog
	if start_face_detect is False:
		return 
	if start_face_recog is True:
		start_face_recog = False
		btn_face_recog.set('开始识别人脸')
	else:
		start_face_recog = True
		btn_face_recog.set('停止识别人脸')


tk.Label(window,text = "人脸检测与识别系统",font=('Arial', 30),pady = 20).pack()

canvas=tk.Canvas(window,width=600,height=600)
canvas.pack()

panel1 = tk.Frame(window)
panel1.pack()
tk.Button(panel1,text='剪切人脸', font=('Arial', 20), padx=30, pady=10, command=save_face).grid(row = 0, column = 0)
tk.Button(panel1,textvariable=btn_face_detect, font=('Arial', 20), padx=30, pady=10, command=face_detection).grid(row = 0, column = 1)
tk.Button(panel1,textvariable=btn_face_recog, font=('Arial', 20), padx=30, pady=10, command=face_recognition).grid(row = 0, column = 2)

vedio_loop()

window.mainloop()
video_capture.release()