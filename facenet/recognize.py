import os
import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import sys
from facenet import facenet
from facenet.knn import knn

class Facenet_Recognize(object):
	def __init__(self,model_path,image_size,npz_file):
		
		# tf.Graph().as_default()
		self.sess = tf.Session()
		facenet.load_model(model_path)
		self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		self.embedding_size = self.embeddings.get_shape()[1]

		data_train = np.load(npz_file)
		self.x_train = data_train['x_train']
		self.y_train = data_train['y_train']
		self.image_size = image_size

	def predict(self,img):
		img = facenet.prewhiten(img)
		emb_array = np.zeros((1,self.embedding_size))
		emb_array[0,:] = self.sess.run(self.embeddings, 
			feed_dict={self.images_placeholder:img.reshape(-1,self.image_size,self.image_size,3),
			self.phase_train_placeholder:False})[0]
		return knn(emb_array,self.x_train,self.y_train,1) 

