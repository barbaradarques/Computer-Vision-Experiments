import os
import numpy as np
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import output2file as o2f
import svm
import boxplot

def main1():
	np.random.seed(1) # a fixed seed guarantees results reproducibility 
	start_time = time.time()
	cnn = VGG16(weights='imagenet')
	layers_names = [layer.name for layer in cnn.layers[-8:-1] if layer.name[-4:] != 'pool'] # << excludes the first and last layers(input and predictions)
	datasets_path = '/home/DADOS1/esouza/Datasets/classified/'
	datasets_names = ['17flowers', 'coil-20', 'corel-1000',  'CUB_200_2011', 'tropical_fruits1400']
	
	for dataset_name in datasets_names:
		preprocessed_imgs, imgs_names, imgs_classes = o2f.batch_preprocessing(datasets_path, dataset_name)
		layers_outputs = o2f.get_layers_outputs(cnn, layers_names, preprocessed_imgs) 

		# saves the outputs all the layers, except for the input and prediction layers
		for i in range(len(layers_names)):
			o2f.save_data('outputs/' + dataset_name +'/' + layers_names[i] +'.txt',
				imgs_names, layers_outputs[i], imgs_classes) # [0] <<<<<< generalize later  

	print("\n\n\n\nExecution time: %s seconds.\n\n\n\n" % (time.time() - start_time))

def main2(): 
	start_time = time.time()
	cnn = VGG16(weights='imagenet')
	layers_names = [layer.name for layer in cnn.layers[-8:-1] if layer.name[-4:] != 'pool'] # << excludes the first and last layers(input and predictions)
	datasets_names = ['17flowers', 'coil-20', 'corel-1000',  'CUB_200_2011', 'tropical_fruits1400']
	for dataset_name in datasets_names:
		for layer_name in layers_names:
			test_linear_SVC_params(dataset_name, layer_name)
			test_rbf_SVC_params(dataset_name, layer_name, 1)
			test_poly_SVC_params(dataset_name, layer_name, 1)

	print("\n\nExecution time: %s seconds.\n\n" % (time.time() - start_time))

def  main3():
	cnn = VGG16(weights='imagenet')
	layers_names = [layer.name for layer in cnn.layers[-8:-1] if layer.name[-4:] != 'pool'] # << excludes the first and last layers(input and predictions)
	datasets_names = ['17flowers', 'coil-20', 'corel-1000',  'CUB_200_2011', 'tropical_fruits1400']
	for dataset_name in datasets_names:
		for layer_name in layers_names:
			boxplot.plot_svm_performance(dataset_name, layer_name, show = False)

################################################################################################


print("\n\n\n\nStarting...\n\n\n\n")
	
main1()


