import os, sys
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def recognition_image(img):
	np_image_data = np.asarray(img)
	
	np_final = np.expand_dims(np_image_data,axix=0)
