__author__ = "Barbara Darques"

import output2file as o2f
import tensorflow as tf
import os
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import datasets
import time
import matplotlib.pyplot as sp
from keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from time import gmtime, strftime
import pickle 
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split

import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
from custom_layers import TiedDenseLayer
from keras import backend as K



# tf.losses.sigmoid_cross_entropy

# def main1():
# 	datasets_names = ['tropical_fruits1400']

# 	for dataset_name in datasets_names:	
# 		layer_name = 'block5_conv1'
# 		names, values, classes = o2f.load_data('outputs/' + dataset_name + '/' + layer_name + '.txt', " ")
# 		tsne_data = dim_reduction.load_2d_data(dataset_name, layer_name)
# 		print(names.shape)
# 		print('tsne_data.shape = ', end=' ')
# 		print(tsne_data.shape)
# 		print('classes = ', end=' ')
# 		print(classes)
# 		values_train, values_test, classes_train, classes_test = train_test_split(tsne_data, classes, test_size=0.9, random_state=0)
		
# 		with open('t-sne_performance/' + dataset_name + '-block5_conv1.csv','w') as file:
# 			clf = svm.SVC(kernel = 'linear') # uses default cost = 1.0 and linear kernel because it was the one that performed better in previous tests
# 			scores = cross_val_score(clf, tsne_data, classes, cv = 10)
# 			scores_str = ",".join(str(i) for i in scores)
# 			file.write(scores_str + '\n') # previously str(cost) + ','

# 	print("\n\nExecution time: %s seconds.\n\n" % (time.time() - start_time))



def test_autoencoder(tag, x_train, x_test):
	print("----- test_autoencoder -----")
	model_id = 'normal'
	if not os.path.exists('autoencoder_results/' + model_id):
		os.makedirs('autoencoder_results/' + model_id)

	input_img = Input(shape=(784,))

	encoded = Dense(256, activation='relu')(input_img)
	encoded = Dense(128, activation='relu')(encoded)

	decoded = Dense(256, activation='relu')(encoded)
	decoded = Dense(784, activation='relu')(decoded)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	##########

	encoder = Model(input_img, encoded)

	##########
	encoded_input = Input(shape=(128,))

	deco = autoencoder.layers[-2](encoded_input)
	deco = autoencoder.layers[-1](deco)
	decoder = Model(encoded_input, deco)

	##########

	# TODO: ADAPT OLD COLD TO NEW STRUCTURE! <<<<<
	####

	# csv_logger = CSVLogger('autoencoder_results/' + tag + '_normal_training.log')
	# tb_callback = TensorBoard(log_dir='./tensorboard_logs/' + tag + '_normal_training',
	# 							histogram_freq=1, write_graph=True, write_images=False)

	history = autoencoder.fit(x_train, x_train,
				epochs=50,
				batch_size=256,
				shuffle=True,
				validation_data=(x_test, x_test)) # callbacks=[csv_logger, tb_callback]

	autoencoder.save('autoencoder_results/normal/' + tag + '_autoencoder.h5')
	encoder.save('autoencoder_results/normal/' + tag + '_encoder.h5')
	decoder.save('autoencoder_results/normal/' + tag + '_decoder.h5')
	with open('autoencoder_results/normal/' + tag + "_history.pckl", 'wb') as pckl:
		pickle.dump(history.history, pckl)
	
	# plot_loss_and_accuracy("MNIST Autoencoder Tradicional", history.history)
	# images, classes = process_mnist()
	# save_encoded_values(tag + '_' + model_id,
						# trained_encoder=encoder, images=images)

def test_conv_autoencoder(tag, x_train, x_test):
	print("\n\n\n----- test_conv_autoencoder -----\n\n\n")

	model_id = 'conv'
	
	if not os.path.exists('autoencoder_results/' + model_id):
		os.makedirs('autoencoder_results/' + model_id)


	if tag == 'mnist' or tag == 'fashion_mnist':
		shape = (28, 28, 1)
	else:
		shape = (64, 64, 3)

	input_img = Input(shape=shape)
	encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	print(encoded._keras_shape)
	encoded = MaxPooling2D((2, 2), padding='same')(encoded)
	print(encoded._keras_shape)
	encoded = Flatten()(encoded)
	print(encoded._keras_shape)
	encoded = Dense(256, activation='relu')(encoded)
	print(encoded._keras_shape)
	encoded = Dense(128, activation='relu')(encoded)
	print(encoded._keras_shape)

	#####
	print('start of the decoder')
	
	decoded = Dense(256, activation='relu')(encoded)
	print(decoded._keras_shape)
	decoded = Dense(int(((16*shape[0]*shape[1])/4)), activation='relu')(decoded)
	print(decoded._keras_shape)
	decoded = Reshape((int(shape[0]/2),int(shape[1]/2),16))(decoded) 
	print(decoded._keras_shape)
	decoded = UpSampling2D((2, 2))(decoded)
	print(decoded._keras_shape)
	decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
	print(decoded._keras_shape)
	decoded = Conv2D(int(shape[2]), (3, 3), activation='sigmoid', padding='same')(decoded)
	print(decoded._keras_shape)
	
	#####

	autoencoder = Model(input_img, decoded)

	autoencoder.summary()
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	##########

	encoder = Model(input_img, encoded)

	##########
	encoded_input = Input(shape=(128,))

	deco = autoencoder.layers[-6](encoded_input)
	deco = autoencoder.layers[-5](deco)
	deco = autoencoder.layers[-4](deco)
	deco = autoencoder.layers[-3](deco)
	deco = autoencoder.layers[-2](deco)
	deco = autoencoder.layers[-1](deco)
	decoder = Model(encoded_input, deco)

	history = autoencoder.fit(x_train, x_train,
				epochs=100,
				batch_size=128,
				shuffle=True,
				validation_data=(x_test, x_test))

	autoencoder.save('autoencoder_results/' + model_id + '/' + tag + '_autoencoder.h5')
	encoder.save('autoencoder_results/' + model_id + '/' + tag + '_encoder.h5')
	decoder.save('autoencoder_results/' + model_id + '/' + tag + '_decoder.h5')

	with open('autoencoder_results/' + model_id + '/' + tag + "_history.pckl", 'wb') as pckl:
		pickle.dump(history.history, pckl)
	
	# plot_loss_and_accuracy("MNIST Autoencoder Convolucional sem Amarração", history.history)
	# images, classes = process_mnist()
	# save_encoded_values(tag + '_' + model_id,
	# 					trained_encoder=encoder, images=images) # save manually because the input was shuffled for training

def flatten_input(x_train, x_test):
	print('--- flatten_input ---')
	# prepare input data
	(x_train, _), (x_test, y_test) = mnist.load_data()

	# normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	print(x_train.shape)
	print(x_test.shape)
	return x_train, x_test


def save_encoded_values(tag, trained_encoder, images):
	print('--- save_encoded_values --- tag: ' + tag)
	output = trained_encoder.predict(images)
	print(output.shape)
	pickle.dump(output, open('autoencoder_results/encoded_outputs/' + tag +".pckl", "wb")) # just for safety

	np.savetxt('autoencoder_results/encoded_outputs/' + tag + ".csv", output, delimiter=",")
	
	

def test_tied_autoencoder(tag, x_train, x_test):
	print("----- test_tied_autoencoder -----")
	model_id = 'tied_transpose_2_128'

	if not os.path.exists('autoencoder_results/' + model_id):
		os.makedirs('autoencoder_results/' + model_id)

	input_img = Input(shape=(784,))

	##### 

	encoding_layer_1 = Dense(256, activation='relu')
	encoding_layer_2 = Dense(128, activation='relu')

	encoded = encoding_layer_1(input_img)
	encoded = encoding_layer_2(encoded)

	#####

	decoded = TiedDenseLayer(output_dim = 256, tied_to = encoding_layer_2, tie_type = 'transpose', activation='relu')(encoded)
	decoded = TiedDenseLayer(output_dim = 784, tied_to = encoding_layer_1, tie_type = 'transpose', activation='relu')(decoded)

	autoencoder = Model(input_img, decoded)
	autoencoder.summary()
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	##########

	encoder = Model(input_img, encoded)

	##########

	encoded_input = Input(shape=(128,))

	deco = autoencoder.layers[-2](encoded_input)
	deco = autoencoder.layers[-1](deco)
	decoder = Model(encoded_input, deco)

	##########

	flat_x_train, flat_x_test = flatten_input(x_train, x_test)

	# csv_logger = CSVLogger('autoencoder_results/' + tag + '_normal_training.log')
	# tb_callback = TensorBoard(log_dir='./tensorboard_logs/' + tag + '_normal_training',
	# 							histogram_freq=1, write_graph=True, write_images=False)

	history = autoencoder.fit(flat_x_train, flat_x_train,
				epochs=50,
				batch_size=256,
				shuffle=True,
				validation_data=(flat_x_test, flat_x_test)) # callbacks=[csv_logger, tb_callback]

	
	autoencoder.save('autoencoder_results/' + model_id + '/' + tag + '_autoencoder.h5')
	encoder.save('autoencoder_results/' + model_id + '/' + tag + '_encoder.h5')
	decoder.save('autoencoder_results/' + model_id + '/' + tag + '_decoder.h5')

	with open('autoencoder_results/' + model_id + '/' + tag + "_history.pckl", 'wb') as pckl:
		pickle.dump(history.history, pckl)
	
	# plot_loss_and_accuracy("MNIST Autoencoder 2 Camadas de Codificação Amarradas por Transposição", history.history)
	# images, classes = process_mnist()
	# save_encoded_values(tag + '_' + model_id,
						# trained_encoder=encoder, images=images)


def test_inverse_tied_autoencoder(tag, x_train, x_test):
	print("----- test_inverse_tied_autoencoder -----")
	model_id = 'tied_inverse_2_128_extended'

	if not os.path.exists('autoencoder_results/' + model_id):
		os.makedirs('autoencoder_results/' + model_id)

	input_img = Input(shape=(784,))
	
	##### 

	encoding_layer_1 = Dense(256)
	encoding_layer_2 = Dense(128)

	encoded = encoding_layer_1(input_img)
	encoded = encoding_layer_2(encoded)

	#####
	
	decoded = TiedDenseLayer(output_dim = 256, tied_to = encoding_layer_2, tie_type = 'inverse', activation='relu')(encoded) 
	decoded = TiedDenseLayer(output_dim = 784, tied_to = encoding_layer_1, tie_type = 'inverse', activation='relu')(decoded)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	##########

	encoder = Model(input_img, encoded)

	##########

	encoded_input = Input(shape=(128,))

	deco = autoencoder.layers[-2](encoded_input)
	deco = autoencoder.layers[-1](deco)
	decoder = Model(encoded_input, deco)

	##########

	flat_x_train, flat_x_test = flatten_input(x_train, x_test)

	# csv_logger = CSVLogger('autoencoder_results/' + tag + '_normal_training.log')
	# tb_callback = TensorBoard(log_dir='./tensorboard_logs/' + tag + '_normal_training',
	# 							histogram_freq=1, write_graph=True, write_images=False)

	history = autoencoder.fit(flat_x_train, flat_x_train,
				epochs=100,
				batch_size=256,
				shuffle=True,
				validation_data=(flat_x_test, flat_x_test)) # callbacks=[csv_logger, tb_callback]

	
	autoencoder.save('autoencoder_results/' + model_id + '/' + tag + '_autoencoder.h5')
	encoder.save('autoencoder_results/' + model_id + '/' + tag + '_encoder.h5')
	decoder.save('autoencoder_results/' + model_id + '/' + tag + '_decoder.h5')

	with open('autoencoder_results/' + model_id + '/' + tag + "_history.pckl", 'wb') as pckl:
		pickle.dump(history.history, pckl)
	
	# plot_loss_and_accuracy("MNIST Autoencoder 2 Camadas de Codificação Amarradas por Inversas Aproximadas - Treinamento Extendido", history.history)
	# images, classes = process_mnist()
	# save_encoded_values(tag + '_' + model_id,
	# 					trained_encoder=encoder, images=images) # save manually because the inputs have been shuffled for training


def test_only_dense_tied_conv_autoencoder(tag, x_train, x_test):
	print("\n\n\n----- test_only_dense_tied_conv_autoencoder -----\n\n\n")

	model_id = 'only_dense_tied_conv'
	
	if not os.path.exists('autoencoder_results/' + model_id):
		os.makedirs('autoencoder_results/' + model_id)


	if tag == 'mnist' or tag == 'fashion_mnist':
		shape = (28, 28, 1)
	else:
		shape = (64, 64, 3)

	input_img = Input(shape=shape)

	encoding_layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')
	encoding_layer_2 = MaxPooling2D((2, 2), padding='same')
	encoding_layer_3 = Flatten()
	encoding_layer_4 = Dense(256, activation='relu')
	encoding_layer_5 = Dense(128, activation='relu')


	encoded = encoding_layer_1(input_img)
	print(encoded._keras_shape)
	encoded = encoding_layer_2(encoded)
	print(encoded._keras_shape)
	encoded = encoding_layer_3(encoded)
	print(encoded._keras_shape)
	encoded = encoding_layer_4(encoded)
	print(encoded._keras_shape)
	encoded = encoding_layer_5(encoded)
	print(encoded._keras_shape)

	#####
	print('start of the decoder')
	
	decoded = TiedDenseLayer(output_dim = 256, tied_to = encoding_layer_5, tie_type = 'transpose', activation='relu')(encoded)
	print(decoded._keras_shape)
	decoded = TiedDenseLayer(output_dim = int(((16*shape[0]*shape[1])/4)), tied_to = encoding_layer_4, tie_type = 'transpose', activation='relu')(decoded)
	print(decoded._keras_shape)
	decoded = Reshape((int(shape[0]/2),int(shape[1]/2),16))(decoded) 
	print(decoded._keras_shape)
	decoded = UpSampling2D((2, 2))(decoded)
	print(decoded._keras_shape)
	decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
	print(decoded._keras_shape)
	decoded = Conv2D(int(shape[2]), (3, 3), activation='sigmoid', padding='same')(decoded)
	print(decoded._keras_shape)
	
	#####

	autoencoder = Model(input_img, decoded)

	autoencoder.summary()
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	##########

	encoder = Model(input_img, encoded)

	##########
	encoded_input = Input(shape=(128,))

	deco = autoencoder.layers[-6](encoded_input)
	deco = autoencoder.layers[-5](deco)
	deco = autoencoder.layers[-4](deco)
	deco = autoencoder.layers[-3](deco)
	deco = autoencoder.layers[-2](deco)
	deco = autoencoder.layers[-1](deco)
	decoder = Model(encoded_input, deco)

	history = autoencoder.fit(x_train, x_train,
				epochs=50,
				batch_size=128,
				shuffle=True,
				validation_data=(x_test, x_test))

	autoencoder.save('autoencoder_results/' + model_id + '/' + tag + '_autoencoder.h5')
	encoder.save('autoencoder_results/' + model_id + '/' + tag + '_encoder.h5')
	decoder.save('autoencoder_results/' + model_id + '/' + tag + '_decoder.h5')

	with open('autoencoder_results/' + model_id + '/' + tag + "_history.pckl", 'wb') as pckl:
		pickle.dump(history.history, pckl)
	
	# plot_loss_and_accuracy("MNIST Autoencoder Convolucional com Camadas Densas Amarradas por Transposição", history.history)
	# images, classes = process_mnist()
	# save_encoded_values(tag + '_' + model_id,
	# 					trained_encoder=encoder, images=images) # save manually because the input was shuffled for training




def test_only_dense_inverse_tied_autoencoder(tag, x_train, x_test):
	print("\n\n\n----- test_only_dense_inverse_tied_autoencoder -----\n\n\n")

	model_id = 'only_dense_inverse_tied'
	
	if not os.path.exists('autoencoder_results/' + model_id):
		os.makedirs('autoencoder_results/' + model_id)


	if tag == 'mnist' or tag == 'fashion_mnist':
		shape = (28, 28, 1)
	else:
		shape = (64, 64, 3)

	input_img = Input(shape=shape)

	encoding_layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')
	encoding_layer_2 = MaxPooling2D((2, 2), padding='same')
	encoding_layer_3 = Flatten()
	encoding_layer_4 = Dense(256, activation='relu')
	encoding_layer_5 = Dense(128, activation='relu')


	encoded = encoding_layer_1(input_img)
	print(encoded._keras_shape)
	encoded = encoding_layer_2(encoded)
	print(encoded._keras_shape)
	encoded = encoding_layer_3(encoded)
	print(encoded._keras_shape)
	encoded = encoding_layer_4(encoded)
	print(encoded._keras_shape)
	encoded = encoding_layer_5(encoded)
	print(encoded._keras_shape)

	#####
	print('start of the decoder')
	
	decoded = TiedDenseLayer(output_dim = 256, tied_to = encoding_layer_5, tie_type = 'inverse', activation='relu')(encoded)
	print(decoded._keras_shape)
	decoded = TiedDenseLayer(output_dim = int(((16*shape[0]*shape[1])/4)), tied_to = encoding_layer_4, tie_type = 'inverse', activation='relu')(decoded)
	print(decoded._keras_shape)
	decoded = Reshape((int(shape[0]/2),int(shape[1]/2),16))(decoded) 
	print(decoded._keras_shape)
	decoded = UpSampling2D((2, 2))(decoded)
	print(decoded._keras_shape)
	decoded =Conv2D(16, (3, 3), activation='relu', border_mode='same')(decoded)
	print(decoded._keras_shape)
	decoded = Conv2D(int(shape[2]), (3, 3), activation='sigmoid', padding='same')(decoded)
	print(decoded._keras_shape)
	
	#####

	autoencoder = Model(input_img, decoded)

	autoencoder.summary()
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	##########

	encoder = Model(input_img, encoded)

	##########
	encoded_input = Input(shape=(128,))

	deco = autoencoder.layers[-6](encoded_input)
	deco = autoencoder.layers[-5](deco)
	deco = autoencoder.layers[-4](deco)
	deco = autoencoder.layers[-3](deco)
	deco = autoencoder.layers[-2](deco)
	deco = autoencoder.layers[-1](deco)
	decoder = Model(encoded_input, deco)

	history = autoencoder.fit(x_train, x_train,
				epochs=50,
				batch_size=128,
				shuffle=True,
				validation_data=(x_test, x_test))

	autoencoder.save('autoencoder_results/' + model_id + '/' + tag + '_autoencoder.h5')
	encoder.save('autoencoder_results/' + model_id + '/' + tag + '_encoder.h5')
	decoder.save('autoencoder_results/' + model_id + '/' + tag + '_decoder.h5')

	with open('autoencoder_results/' + model_id + '/' + tag + "_history.pckl", 'wb') as pckl:
		pickle.dump(history.history, pckl)
	
	# plot_loss_and_accuracy("MNIST Autoencoder Convolucional com Camadas Densas Amarradas por Inversas Aproximadas", history.history)
	# images, classes = process_mnist()
	# save_encoded_values(tag + '_' + model_id,
	# 					trained_encoder=encoder, images=images) # save manually because the input was shuffled for training


def test_3_conv_layers_autoencoder(tag, x_train, x_test):
	print("\n\n\n----- test_3_conv_layers_autoencoder -----\n\n\n")

	model_id = '3_conv_layers'
	
	if not os.path.exists('autoencoder_results/' + model_id):
		os.makedirs('autoencoder_results/' + model_id)


	if tag == 'mnist' or tag == 'fashion_mnist':
		shape = (28, 28, 1)
	else:
		shape = (64, 64, 1)

	input_img = Input(shape=shape)

	encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	print(encoded._keras_shape)
	
	encoded = MaxPooling2D((2, 2), padding='same')(encoded)
	print(encoded._keras_shape)

	# encoded = BatchNormalization()(encoded)

	# encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
	# print(encoded._keras_shape)
	
	# encoded = MaxPooling2D((2, 2), padding='same')(encoded)
	# print(encoded._keras_shape)
	
	encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
	print(encoded._keras_shape)
	
	encoded = MaxPooling2D((2, 2), padding='same')(encoded)
	print(encoded._keras_shape)
	
	encoded = Flatten()(encoded)
	print(encoded._keras_shape)
	
	encoded = Dense(256, activation='relu')(encoded)
	print(encoded._keras_shape)
	
	encoded = Dense(128, activation='relu')(encoded)
	print(encoded._keras_shape)

	#####
	print('start of the decoder')
	
	decoded = Dense(256, activation='relu')(encoded)
	print(decoded._keras_shape)
	
	decoded = Dense(int(((16*shape[0]*shape[1])/4)), activation='relu')(decoded)
	print(decoded._keras_shape)
	
	decoded = Reshape((int(shape[0]/2),int(shape[1]/2),16))(decoded) 
	print(decoded._keras_shape)

	decoded = UpSampling2D((2, 2))(decoded)
	print(decoded._keras_shape)
	
	decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
	print(decoded._keras_shape)
	
	decoded = Conv2D(int(shape[2]), (3, 3), activation='sigmoid', padding='same')(decoded)
	print(decoded._keras_shape)
	
	#####

	autoencoder = Model(input_img, decoded)

	autoencoder.summary()
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	##########

	encoder = Model(input_img, encoded)

	##########
	encoded_input = Input(shape=(128,))

	deco = autoencoder.layers[-6](encoded_input)
	deco = autoencoder.layers[-5](deco)
	deco = autoencoder.layers[-4](deco)
	deco = autoencoder.layers[-3](deco)
	deco = autoencoder.layers[-2](deco)
	deco = autoencoder.layers[-1](deco)
	decoder = Model(encoded_input, deco)

	history = autoencoder.fit(x_train, x_train,
				epochs=100,
				batch_size=128,
				shuffle=True,
				validation_data=(x_test, x_test))

	autoencoder.save('autoencoder_results/' + model_id + '/' + tag + '_autoencoder.h5')
	encoder.save('autoencoder_results/' + model_id + '/' + tag + '_encoder.h5')
	decoder.save('autoencoder_results/' + model_id + '/' + tag + '_decoder.h5')

	with open('autoencoder_results/' + model_id + '/' + tag + "_history.pckl", 'wb') as pckl:
		pickle.dump(history.history, pckl)
	
	# plot_loss_and_accuracy("MNIST Autoencoder Convolucional sem Amarração", history.history)
	# images, classes = process_mnist()
	# save_encoded_values(tag + '_' + model_id,
	# 					trained_encoder=encoder, images=images) # save manually because the input was shuffled for training


def test_only_dense_tied_2_layer_conv_autoencoder(tag, x_train, x_test):
	print("\n\n\n----- test_only_dense_tied_conv_autoencoder -----\n\n\n")

	model_id = 'only_dense_tied_conv'
	
	if not os.path.exists('autoencoder_results/' + model_id):
		os.makedirs('autoencoder_results/' + model_id)


	if tag == 'mnist' or tag == 'fashion_mnist':
		shape = (28, 28, 1)
	else:
		shape = (64, 64, 3)

	input_img = Input(shape=shape)

	encoding_layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')
	encoding_layer_2 = MaxPooling2D((2, 2), padding='same')
	encoding_layer_3 = Flatten()
	encoding_layer_4 = Dense(256, activation='relu')
	encoding_layer_5 = Dense(128, activation='relu')


	encoded = encoding_layer_1(input_img)
	print(encoded._keras_shape)
	encoded = encoding_layer_2(encoded)
	print(encoded._keras_shape)
	encoded = encoding_layer_3(encoded)
	print(encoded._keras_shape)
	encoded = encoding_layer_4(encoded)
	print(encoded._keras_shape)
	encoded = encoding_layer_5(encoded)
	print(encoded._keras_shape)

	#####
	print('start of the decoder')
	
	decoded = TiedDenseLayer(output_dim = 256, tied_to = encoding_layer_5, tie_type = 'transpose', activation='relu')(encoded)
	print(decoded._keras_shape)
	decoded = TiedDenseLayer(output_dim = int(((16*shape[0]*shape[1])/4)), tied_to = encoding_layer_4, tie_type = 'transpose', activation='relu')(decoded)
	print(decoded._keras_shape)
	decoded = Reshape((int(shape[0]/2),int(shape[1]/2),16))(decoded) 
	print(decoded._keras_shape)
	decoded = UpSampling2D((2, 2))(decoded)
	print(decoded._keras_shape)
	decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
	print(decoded._keras_shape)
	decoded = Conv2D(int(shape[2]), (3, 3), activation='sigmoid', padding='same')(decoded)
	print(decoded._keras_shape)
	
	#####

	autoencoder = Model(input_img, decoded)

	autoencoder.summary()
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	##########

	encoder = Model(input_img, encoded)

	##########
	encoded_input = Input(shape=(128,))

	deco = autoencoder.layers[-6](encoded_input)
	deco = autoencoder.layers[-5](deco)
	deco = autoencoder.layers[-4](deco)
	deco = autoencoder.layers[-3](deco)
	deco = autoencoder.layers[-2](deco)
	deco = autoencoder.layers[-1](deco)
	decoder = Model(encoded_input, deco)

	history = autoencoder.fit(x_train, x_train,
				epochs=50,
				batch_size=128,
				shuffle=True,
				validation_data=(x_test, x_test))

	autoencoder.save('autoencoder_results/' + model_id + '/' + tag + '_autoencoder.h5')
	encoder.save('autoencoder_results/' + model_id + '/' + tag + '_encoder.h5')
	decoder.save('autoencoder_results/' + model_id + '/' + tag + '_decoder.h5')

	with open('autoencoder_results/' + model_id + '/' + tag + "_history.pckl", 'wb') as pckl:
		pickle.dump(history.history, pckl)
	
	# plot_loss_and_accuracy("MNIST Autoencoder Convolucional com Camadas Densas Amarradas por Transposição", history.history)
	# images, classes = process_mnist()
	# save_encoded_values(tag + '_' + model_id,
	# 					trained_encoder=encoder, images=images) # save manually because the input was shuffled for training


def plot_loss_and_accuracy(tag, history):
	# # TODO REMOVE THIS <<<<<
	# history = pickle.load(open('autoencoder_results/normal/mnist_history.pckl', 'rb'))

	fig = plt.figure(figsize=(15.0, 10.0)) # 1,
	sp = fig.add_subplot(111)
	# summarize history for loss
	sp.plot(history['loss'])
	sp.plot(history['val_loss'])
	sp.title.set_text(tag + '\n')
	sp.set_ylabel('erro')
	sp.set_xlabel('época')
	sp.legend(['treinamento', 'teste'], loc='upper left')
	plt.show()

	fig.tight_layout()

	fig.savefig('autoencoders_acc_plots/' + tag.lower().replace(' ','_') + '.png', dpi=100, bbox_inches='tight')
	del fig


def load_training_history(filename):
	# filename = 'autoencoder_results/normal/mnist_history.pckl'
	return pickle.load(open(filename, 'rb'))


def load_trained_model(filename):
	return load_model(filename, custom_objects={'TiedDenseLayer': TiedDenseLayer})


def main1():
	(x_train, _), (x_test, y_test) = fashion_mnist.load_data()

	flat_x_train, flat_x_test = flatten_input(x_train, x_test)

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

	# test_autoencoder('fashion_mnist', flat_x_train, flat_x_test)
	# test_only_dense_inverse_tied_autoencoder('fashion_mnist', x_train, x_test)
	test_conv_autoencoder('fashion_mnist', x_train, x_test)
	# test_tied_autoencoder('fashion_mnist', flat_x_train, flat_x_test)
	# test_inverse_tied_autoencoder('fashion_mnist', flat_x_train, flat_x_test)
	# test_only_dense_tied_conv_autoencoder('fashion_mnist', x_train, x_test)

def main10():
	(x_train, _), (x_test, y_test) = mnist.load_data()
	print("x_train shape:", x_train.shape)
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
	test_only_dense_inverse_tied_autoencoder('fashion_mnist', x_train, x_test)


def main2():	
	history = load_training_history('autoencoder_results/only_dense_inverse_tied/fashion_mnist_history.pckl')
	plot_loss_and_accuracy("Fashion-MNIST - Autoencoder Convolucional com Camadas Densas Amarradas por Inversas Aproximadas", history)
	
	history = load_training_history('autoencoder_results/normal/fashion_mnist_history.pckl')
	plot_loss_and_accuracy("Fashion-MNIST - Autoencoder Tradicional sem Amarração de Pesos" , history)
	
	history = load_training_history('autoencoder_results/conv/fashion_mnist_history.pckl')
	plot_loss_and_accuracy( "Fashion-MNIST - Autoencoder Convolucional sem Amarração de Pesos", history)
	
	history = load_training_history('autoencoder_results/tied_transpose_2_128/fashion_mnist_history.pckl')
	plot_loss_and_accuracy("Fashion-MNIST - Autoencoder Tradicional com Camadas Amarradas por Transposição" , history)
	
	history = load_training_history('autoencoder_results/tied_inverse_2_128_extended/fashion_mnist_history.pckl')
	plot_loss_and_accuracy("Fashion-MNIST - Autoencoder Tradicional com Camadas Amarradas por Inversas Aproximadas - Treinamento Estendido" , history)
	
	history = load_training_history('autoencoder_results/only_dense_tied_conv/fashion_mnist_history.pckl')
	plot_loss_and_accuracy("Fashion-MNIST - Autoencoder Convolucional com Camadas Densas Amarradas por Transposição" , history)

def main3():
	(x_train, _), (x_test, y_test) = mnist.load_data()
	x_train, x_test = flatten_input(x_train, x_test)
	test_inverse_tied_autoencoder('mnist', x_train, x_test)

def main4():
	encoder = load_trained_model('autoencoder_results/conv/mnist_encoder.h5')
	images, classes = process_mnist()
	images = images.astype('float32') / 255.
	images = np.reshape(images, (len(images), 28, 28, 1))
	
	save_encoded_values('mnist_conv',
						trained_encoder=encoder, images=images)

def main5():
	datasets_path = '/home/DADOS1/esouza/Datasets/classified/'
	datasets_names = ['17flowers', 'coil-20', 'corel-1000', 'tropical_fruits1400']
	# datasets_names = ['tropical_fruits1400']
	# datasets_path = ''
	for dataset_name in datasets_names:
		preprocessed_imgs, imgs_names, imgs_classes = o2f.batch_preprocessing(datasets_path, dataset_name, target_size=64)
		print(preprocessed_imgs.shape)
		x_train, x_test = train_test_split(preprocessed_imgs)
		x_train = x_train.astype('float32') / 255.
		x_test = x_test.astype('float32') / 255.
		print("input was shuffled...")
		test_conv_autoencoder(dataset_name, x_train, x_test)


def process_mnist():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	images = np.concatenate([x_train, x_test])
	classes = np.concatenate([y_train, y_test])
	return images, classes



def main6():
	# datasets_path = '/home/DADOS1/esouza/Datasets/classified/'
	# datasets_names = ['17flowers', 'coil-20', 'corel-1000', 'tropical_fruits1400']
	datasets_names = ['tropical_fruits1400']
	datasets_path = ''
	for dataset_name in datasets_names:
		preprocessed_imgs, imgs_names, imgs_classes = o2f.centered_square_batch_preprocessing(datasets_path, dataset_name)

		# print(preprocessed_imgs[0][0])
		x_train, x_test = train_test_split(preprocessed_imgs)
		x_train = x_train.astype('float32') / 255.
		x_test = x_test.astype('float32') / 255.
		print("input was shuffled...")
		test_3_conv_layers_autoencoder(dataset_name, x_train, x_test)


if __name__ == '__main__':
	print("\n\n\n\nStarting...\n\n\n\n")
	np.random.seed(1) # a fixed seed guarantees results reproducibility 
	start_time = time.time()

	main6()
	# print(K.image_data_format())

	print("\n\nExecution time: %s seconds.\n\n" % (time.time() - start_time))
	
	


