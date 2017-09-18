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

	print("aqui")
	cnn = VGG16(weights='imagenet')
	# cnn.summary()
	layer_names = [layer.name for layer in cnn.layers[1:-1]] # << excludes the first and last layers(input and predictions)
	# print(len(layer_names))
	dataset_name = 'Produce_1400'
	preprocessed_imgs, imgs_names, imgs_classes = o2f.batch_preprocessing(dataset_name)

	layers_outputs = o2f.get_layers_outputs(cnn, layer_names, preprocessed_imgs) 

	# print('Predicted:', decode_predictions(layers_outputs, top=5))

	# saves the outputs all the layers, except for the input and prediction layers
	for i in range(len(layer_names)):
		o2f.save_data('outputs/' + dataset_name +'/' + layer_names[i] +'.txt',
			imgs_names, layers_outputs[i], imgs_classes) 


	# arr = np.arange(36).reshape((3,2,2,3))
	# print(arr.reshape(3,12))

	# o2f.save_data('teste-fc1.txt', imgs_names[:5], layers_outputs[0], imgs_classes[:5])

	# o2f.save_data('teste-fc2.txt', imgs_names[:5], layers_outputs[1], imgs_classes[:5])

	# names, values, classes = o2f.load_data("teste-fc1.txt", " ")

	# arr = np.array([[[1,2,3],[4,5, 6]],[[7,8,9], [10, 11, 12]]])
	# arr = np.array([1,2,3])
	# print(len(arr.shape))
	# print(arr)
	# arr = arr.ravel()
	# print(arr)

	print("\n\n\n\nExecution time: %s seconds.\n\n\n\n" % (time.time() - start_time))

def main2(): 
	start_time = time.time()

	layer_output_file = "produce-fc1.txt"
	#test_linear_SVC_params('Produce_1400', layer_output_file)
	test_rbf_SVC_params('Produce_1400', layer_output_file, 1)
	#test_poly_SVC_params('Produce_1400', layer_output_file, 1)

	print("\n\nExecution time: %s seconds.\n\n" % (time.time() - start_time))

def  main3():
	dataset_name = 'Produce_1400'
	layer_name = 'fc1'
	boxplot.plot_svm_performance(dataset_name, layer_name)

################################################################################################
print("antes")
main1()


