__author__ = "Barbara Darques"

import output2file as o2f
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import datasets
import time
import matplotlib.pyplot as sp
from keras.datasets import mnist
from time import gmtime, strftime
import pickle 
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, TensorBoard
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from custom_layers import TiedDenseLayer

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

	input_img = Input(shape=(784,))

	encoded = Dense(256, activation='relu')(input_img)
	encoded = Dense(128, activation='relu')(encoded)

	decoded = Dense(256, activation='relu')(encoded)
	decoded = Dense(784, activation='relu')(decoded)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['acc'])

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

	autoencoder.save('autoencoder_results/normal/' + tag + '_autoencoder.h5')
	encoder.save('autoencoder_results/normal/' + tag + '_encoder.h5')
	decoder.save('autoencoder_results/normal/' + tag + '_decoder.h5')
	with open('autoencoder_results/normal/' + tag + "_history.pckl", 'wb') as pckl:
		pickle.dump(history.history, pckl)
	
	plot_loss_and_accuracy("MNIST Autoencoder Tradicional", history.history)
	save_encoded_values(tag + '_normal', preprocessed_x_train=flat_x_train,
						trained_encoder=encoder, preprocessed_x_test=flat_x_test)


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



def save_encoded_values(tag, preprocessed_x_train, trained_encoder, preprocessed_x_test=None):
	print('--- save_encoded_values --- tag: ' + tag)
	output = trained_encoder.predict(preprocessed_x_train)
	print(output.shape)
	pickle.dump(output, open('autoencoder_results/encoded_outputs/' + tag +"_x_train.pckl", "wb")) # just for safety

	np.savetxt('autoencoder_results/encoded_outputs/' + tag + "_x_train.csv", output, delimiter=",")
	if preprocessed_x_test is not None:
		output = trained_encoder.predict(preprocessed_x_test)
		print(output.shape)
		np.savetxt('autoencoder_results/encoded_outputs/' + tag + "_x_test.csv", output, delimiter=",")

	

def test_tied_autoencoder(tag, x_train, x_test):
	print("----- test_tied_autoencoder -----")
	model_id = 'tied_transpose_2_128'

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
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['acc'])

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
	
	plot_loss_and_accuracy("MNIST Autoencoder 2 Camadas de Codificação Amarradas por Transposição", history.history)
	save_encoded_values(tag + '_' + model_id, preprocessed_x_train=flat_x_train,
						trained_encoder=encoder, preprocessed_x_test=flat_x_test)

def test_inverse_tied_autoencoder(tag, x_train, x_test):
	print("----- test_inverse_tied_autoencoder -----")
	model_id = 'tied_inverse_2_128_extended'

	input_img = Input(shape=(784,))
	
	##### 

	encoding_layer_1 = Dense(256, activation='relu')
	encoding_layer_2 = Dense(128, activation='relu')

	encoded = encoding_layer_1(input_img)
	encoded = encoding_layer_2(encoded)

	#####
	
	decoded = TiedDenseLayer(output_dim = 256, tied_to = encoding_layer_2, tie_type = 'inverse', activation='relu')(encoded)
	decoded = TiedDenseLayer(output_dim = 784, tied_to = encoding_layer_1, tie_type = 'inverse', activation='relu')(decoded)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['acc'])

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
	
	plot_loss_and_accuracy("MNIST Autoencoder 2 Camadas de Codificação Amarradas por Inversas Aproximadas\n(Treinamento Extendido)", history.history)
	save_encoded_values(tag + '_' + model_id, preprocessed_x_train=flat_x_train,
						trained_encoder=encoder, preprocessed_x_test=flat_x_test)



def test_conv_autoencoder():
	print("----- test_conv_autoencoder -----")
	model_id = 'inverse_tied_transpose_1_128'

	input_img = Input(shape=(784,))

	encoding_layer = Dense(128, activation='relu')
	encoded = encoding_layer(input_img)
	# encoded = Dense(128, activation='relu')(encoded)

	decoded = TiedDenseLayer(output_dim = 784, tied_to = encoding_layer, tie_type = 'inverse', activation='relu')(encoded)
	# decoded = Dense(784, activation='relu')(decoded)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['acc'])

	##########

	encoder = Model(input_img, encoded)

	##########
	encoded_input = Input(shape=(128,))

	deco = autoencoder.layers[-1](encoded_input)
	# deco = autoencoder.layers[-1](deco)
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
	
	plot_loss_and_accuracy("MNIST Autoencoder 1 Camada de Codificação Amarrada por Inversas Aproximadas", history.history)
	save_encoded_values(tag + '_' + model_id, preprocessed_x_train=flat_x_train,
						trained_encoder=encoder, preprocessed_x_test=flat_x_test)




def test_conv_tied_autoencoder():
	pass



def plot_loss_and_accuracy(tag, history):
	# # TODO REMOVE THIS <<<<<
	# history = pickle.load(open('autoencoder_results/normal/mnist_history.pckl', 'rb'))

	fig = plt.figure(figsize=(15.0, 10.0)) # 1,
	plt.title(tag + '\n\n', fontsize=16)
	plt.axis('off')
	# list all data in history
	print(history.keys())
	# summarize history for accuracy
	sp = fig.add_subplot(211)
	
	sp.plot(history['acc'])
	sp.plot(history['val_acc'])
	sp.title.set_text('Acurácia do Modelo')
	sp.set_ylabel('acurácia')
	sp.set_xlabel('época')
	sp.legend(['treinamento', 'teste'], loc='upper left')

	sp = fig.add_subplot(212)
	# summarize history for loss
	sp.plot(history['loss'])
	sp.plot(history['val_loss'])
	sp.title.set_text('Função de Custo')
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
	(x_train, _), (x_test, y_test) = mnist.load_data()
	x_train, x_test = flatten_input(x_train, x_test)
	test_tied_autoencoder('mnist', x_train, x_test)



def main2():	

	plot_loss_and_accuracy("MNIST Autoencoder Tradicional", None)

def main3():
	(x_train, _), (x_test, y_test) = mnist.load_data()
	x_train, x_test = flatten_input(x_train, x_test)
	test_inverse_tied_autoencoder('mnist', x_train, x_test)

def main4():
	encoder = load_trained_model('tied_inverse_1_128/mnist_encoder.h5')
	save_encoded_values('mnist_tied_inverse_1_128', preprocessed_x_train=flat_x_train,
						trained_encoder=encoder, preprocessed_x_test=flat_x_test)

if __name__ == '__main__':
	print("\n\n\n\nStarting...\n\n\n\n")
	np.random.seed(1) # a fixed seed guarantees results reproducibility 
	start_time = time.time()

	main3()

	print("\n\nExecution time: %s seconds.\n\n" % (time.time() - start_time))
	
	


