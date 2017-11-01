'''
	This module feeds images to a pre-trained CNN
	and saves the output of selected layers in the following format:
		<image name> <output values> <image ground truth label>

'''
__author__ = "Barbara Darques"

import os
import numpy as np
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


def save_data(filename, imgs_names, output_values, imgs_classes, **kwargs):
	'''
		Saves the following pattern to the given file:
		<image name> <output values> <image ground truth label>
	'''

	# <<<<<<<<<<<< 

	if(len(output_values.shape) == 4): # if the output is multidimensional, it gets flattened
		output_values = output_values.reshape(output_values.shape[0], 
			output_values.shape[1] * output_values.shape[2] * output_values.shape[3])

	# <<<<<<<<<<<<
	# print(filename[len('outputs/Produce_1400/'):] + ':');
	# print('output_values shape = ' + str(output_values.shape))
	# print('imgs_names shape = ' + str(len(imgs_names)))
	# # <<<<<<<<<<<<

	concat1 = np.r_['1,2,0', imgs_names, output_values]
	concat2 = np.r_['1,2,0', concat1, imgs_classes]
	
	if ('append' not in kwargs) or ('append' in kwargs and kwargs['append'] == False): 
		# deletes file if it already exists
		try:
			os.remove(filename)
		except OSError:
			pass
	else:
		print('appending to the old file...')

	with open(filename, 'ab') as output_file: # append to the end of the file
		np.savetxt(output_file, concat2, fmt = '%s')

def load_data(filename, separator):
	'''
		Reads the following pattern from the given file:
		<image name> <output values> <image ground truth label>
		Returns 3 arrays: 'names', 'values' and 'classes'
	'''
	data = []
	with open(filename,'r') as input_file:
		for line in input_file.readlines():
			data.append(line.replace('\n','').split(separator))
	data = np.array(data)
	classes = data[:, -1]
	names = data[:, 0]
	values = data[:, 1:-1].astype('float64')
	return names, values, classes

def batch_preprocessing(datasets_path, dataset_name, **kwargs): 
	print(datasets_path + dataset_name)
	subdirs = next(os.walk(datasets_path + dataset_name))[1] # returns all the subdirectories inside Produce_1400
	# print(subdirs)
	all_imgs = []
	all_imgs_names = []
	all_imgs_classes = []
	if 'start_subdir' in kwargs and 'end_subdir' in kwargs:
		start = kwargs['start_subdir']
		end = kwargs['end_subdir']
		subdirs = subdirs[start:end]
	for subdir in subdirs:
		imgs_names = [subdir+'/'+img for img in os.listdir(datasets_path + dataset_name +'/'+subdir) if img.endswith('.jpg') or img.endswith('.png')]
		imgs_classes = np.empty(len(imgs_names), dtype=np.str)
		imgs_classes.fill(subdir) # as the classes are separated by subdirectories, class name = subdir name
		all_imgs_classes.extend(imgs_classes) # <<<<<<<<<<<<< 
		all_imgs_names.extend(imgs_names) # <<<<<<<<<<<<< 
		for img_name in imgs_names: # <<<<<<<<<<<<< 
			img = image.load_img(datasets_path + dataset_name +'/'+img_name, target_size=(224, 224))
			img = image.img_to_array(img)
			all_imgs.append(img) # note that extend and append are different!

	preprocessed_imgs = preprocess_input(np.array(all_imgs))
	return preprocessed_imgs, all_imgs_names, all_imgs_classes

def get_layers_outputs(cnn, layers_names, preprocessed_imgs):
	layers = [cnn.get_layer(layer_name).output for layer_name in layers_names]
	new_model = Model(inputs=cnn.input, outputs=layers)
	return new_model.predict(preprocessed_imgs)



