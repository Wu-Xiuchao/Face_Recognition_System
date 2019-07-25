import os
import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import sys
sys.path.append('..')
from facenet import facenet
from glob import glob

image_size = 160
modeldir = '../facenet/models/20170512-110547.pb'

tf.Graph().as_default()
sess = tf.Session()

facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

scaled_reshape = []

face_data = '../face_data'
file_list = glob(os.path.join(face_data,'*.jpg'))

x_train = []
y_train = []
for i,image_name in enumerate(file_list):
	image = scipy.misc.imread(image_name, mode='RGB')
	image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
	image = facenet.prewhiten(image)
	scaled_reshape.append(image.reshape(-1,image_size,image_size,3))
	emb_array2 = np.zeros((1, embedding_size))
	emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[i], phase_train_placeholder: False })[0]
	x_train.append(emb_array2[0])
	y_train.append(image_name.split('/')[-1])

x_train = np.array(x_train)

np.savez('../facenet/data.npz',x_train=x_train,y_train=y_train) # [num_of_image * 128]

print('Finished update!')