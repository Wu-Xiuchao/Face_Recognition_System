import cv2
import os
import sys

save_path = 'videos'
video_path = sys.argv[1]
down_sample_rate = int(sys.argv[2]) # 1,2,3,4,5....

if os.path.exists(save_path) is False:
	os.mkdir(save_path)

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

out = cv2.VideoWriter(os.path.join(save_path,'{}.avi'.format(video_path.split('.')[0])),fourcc, 10, (frame_width,frame_height))

count = 0
while True:
	ret, frame = cap.read()
	count += 1
	if ret==True:
		if count%down_sample_rate==0:
			out.write(frame)
	else:
		break
out.release()
cap.release()