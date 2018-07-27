import os
import numpy as np
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from keras.models import Model
import output2file as o2f
from sklearn import svm
import svm_tests
import dim_reduction
import boxplot


def main_cub():
	np.random.seed(1) # a fixed seed guarantees results reproducibility 
	start_time = time.time()
	cnn = VGG16(weights='imagenet')
	layers_names = [layer.name for layer in cnn.layers[-8:-1] if layer.name[-4:] != 'pool'] # << excludes the first and last layers(input and predictions)
	datasets_path = '/home/DADOS1/esouza/Datasets/classified/'
	# datasets_names = ['17flowers', 'coil-20', 'corel-1000',  'CUB_200_2011', 'tropical_fruits1400']
	datasets_names = ['CUB_200_2011']
	for dataset_name in datasets_names:
		for subdir_idx in range(200): # processes each class/subdirectory at a time, because the dataset is too big for being processe all at once 
			preprocessed_imgs, imgs_names, imgs_classes = o2f.batch_preprocessing(datasets_path, dataset_name, start_subdir = subdir_idx, end_subdir = (subdir_idx + 1))
			layers_outputs = o2f.get_layers_outputs(cnn, layers_names, preprocessed_imgs) 

			# saves the outputs all the layers, except for the input and prediction layers
			for i in range(len(layers_names)):
				o2f.save_data('outputs/' + dataset_name +'/' + layers_names[i] +'.txt',
					imgs_names, layers_outputs[i], imgs_classes, append = True) # [0] <<<<<< generalize later  

	print("\n\n\n\nExecution time: %s seconds.\n\n\n\n" % (time.time() - start_time))

def main1():
	np.random.seed(1) # a fixed seed guarantees results reproducibility 
	start_time = time.time()
	cnn = VGG16(weights='imagenet')
	layers_names = [layer.name for layer in cnn.layers[-8:-1] if layer.name[-4:] != 'pool'] # << excludes the first and last layers(input and predictions)
	datasets_path = '/home/DADOS1/esouza/Datasets/classified/'
	# datasets_names = ['17flowers', 'coil-20', 'corel-1000',  'CUB_200_2011', 'tropical_fruits1400']
	datasets_names = ['corel-1000']
	for dataset_name in datasets_names: # <<<<<< temporary
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
	datasets_names = ['CUB_200_2011']
	for dataset_name in datasets_names:
		print('-------')
		print('dataset = ' + dataset_name)
		for layer_name in layers_names:
			print('- layer = ' + layer_name) 
			print('linear')
			svm_tests.test_linear_SVC_params(dataset_name, layer_name)
			print('rbf')
			svm_tests.test_rbf_SVC_params(dataset_name, layer_name, 1)
			print('poly')
			svm_tests.test_poly_SVC_params(dataset_name, layer_name, 1)

	print("\n\nExecution time: %s seconds.\n\n" % (time.time() - start_time))

def  main3():
	cnn = VGG16(weights='imagenet')
	layers_names = [layer.name for layer in cnn.layers[-8:-1] if layer.name[-4:] != 'pool'] # << excludes the first and last layers(input and predictions)
	datasets_names = ['17flowers', 'coil-20', 'corel-1000', 'tropical_fruits1400']
	for dataset_name in datasets_names:
		for layer_name in layers_names:
			boxplot.plot_svm_performance(dataset_name, layer_name, show = False)

def  main4():
	cnn = VGG16(weights='imagenet')
	layers_names = [layer.name for layer in cnn.layers[-8:-1] if (layer.name[-4:] != 'pool') and (layer.name != 'flatten')]
	datasets_names = ['17flowers', 'coil-20', 'corel-1000', 'tropical_fruits1400']
	for dataset_name in datasets_names:
			boxplot.plot_svm_performance_per_dataset(dataset_name, layers_names, show = False)

def main5():
	datasets_names = ['tropical_fruits1400']

	for dataset_name in datasets_names:	
		layer_name = 'block5_conv1'
		names, values, classes = o2f.load_data('outputs/' + dataset_name + '/' + layer_name + '.txt', " ")
		tsne_data = dim_reduction.load_2d_data(dataset_name, layer_name)
		print(names.shape)
		print('tsne_data.shape = ', end=' ')
		print(tsne_data.shape)
		print('classes = ', end=' ')
		print(classes)
		values_train, values_test, classes_train, classes_test = train_test_split(tsne_data, classes, test_size=0.9, random_state=0)
		
		with open('t-sne_performance/' + dataset_name + '-block5_conv1.csv','w') as file:
			clf = svm.SVC(kernel = 'linear') # uses default cost = 1.0 and linear kernel because it was the one that performed better in previous tests
			scores = cross_val_score(clf, tsne_data, classes, cv = 10)
			scores_str = ",".join(str(i) for i in scores)
			file.write(scores_str + '\n') # previously str(cost) + ','

	print("\n\nExecution time: %s seconds.\n\n" % (time.time() - start_time))


if __name__ == '__main__':
	print("\n\n\n\nStarting...\n\n\n\n")
	
	main5()


