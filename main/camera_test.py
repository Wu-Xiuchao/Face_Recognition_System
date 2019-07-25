import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import numpy as np
import cv2,os,re
from facenet.recognize import Facenet_Recognize

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
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 240)
video_capture.set(4, 320)
# fps = video_capture.get(cv2.CAP_PROP_FPS)

# 窗口
frame_W = 640
frame_H = 480
detect_W = 160
detect_H = 120
cv2.namedWindow('Face Recognition',0)
cv2.resizeWindow('Face Recognition',frame_W,frame_H)

# 裁剪最大人脸
def crop_max_face(frame,boxes_c):
	max_index = np.argmax((boxes_c[:,2]-boxes_c[:,0])*(boxes_c[:,3]-boxes_c[:,1]))
	bbox = boxes_c[max_index,:4]
	corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
	pic = frame[corpbbox[1]:corpbbox[3],corpbbox[0]:corpbbox[2]]
	pic = cv2.resize(pic,(image_size,image_size),interpolation=cv2.INTER_CUBIC)
	return pic,corpbbox

name = ['']
ava = np.array([])

while True:
	ret,frame = video_capture.read()
	if ret:

		image = np.array(frame)
		image = cv2.resize(image,(detect_W,detect_H),interpolation=cv2.INTER_CUBIC)
		boxes,_ = mtcnn_detector.detect(image)
		boxes *= (frame_W/detect_W)

		for i in range(boxes.shape[0]):
			bbox = boxes[i,:4]
			corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
			cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),(corpbbox[2], corpbbox[3]), (0,155,255), 2)
			cv2.putText(frame, '_'.join(name), (corpbbox[0], corpbbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)

		if len(ava) > 0:
			frame[:100,:100] = ava

		cv2.imshow('Face Recognition',frame)

		k = cv2.waitKey(1)

		if k == ord('q'):
			break
		elif k == ord(' ') and len(boxes) > 0:
			pic,corpbbox = crop_max_face(frame,boxes)
			path = face_recognize.predict(pic)[0][0]
			ava = cv2.imread('../face_data/'+path)
			ava = cv2.resize(ava,(100,100),interpolation=cv2.INTER_CUBIC)
			name = path.split('.')[0].split('_')[:-1]
		elif k == ord('c') and len(boxes) > 0:
			pic,corpbbox = crop_max_face(frame,boxes)
			cv2.imwrite('../face_data/unnamed_001.jpg',pic)
	
