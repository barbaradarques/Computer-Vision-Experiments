__author__ = "Barbara Darques"

import output2file as o2f
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import datasets
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist
from time import gmtime, strftime

def main1():
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


def test_autoencoder(tag, x_train, x_test):
	print("----- test_autoencoder -----")

	input_img = Input(shape=(784,))

	encoded = Dense(256, activation='relu')(input_img)
	encoded = Dense(128, activation='relu')(encoded)

	decoded = Dense(128, activation='relu')(encoded)
	decoded = Dense(256, activation='relu')(decoded)

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

	(x_train, x_test) = flatten_input(x_train, x_test)

	csv_logger = CSVLogger('autoencoder_results/' + tag + '_normal_training.log')
	history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[csv_logger])

	autoencoder.save('autoencoder_results/' + tag + '_normal.h5')


 

def flatten_input(x_train, x_test)
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



def save_encoded_values(tag, preprocessed_x_train, preprocessed_x_test = None, trained_encoder):
	print('--- save_encoded_values --- tag: ' + tag)
	output = trained_encoder.predict(preprocessed_x_train)
	print(output.shape)
	pickle.dump(output, open("save.pckl", "wb")) # just for safety

	
	numpy.savetxt(tag + "_x_train.csv", output, delimiter=",")
	if preprocessed_x_test is not None:
		output = trained_encoder.predict(preprocessed_x_test)
		print(output.shape)
		numpy.savetxt(tag + "_x_test.csv", output, delimiter=",")

	

def test_tied_autoencoder():
	pass

def test_conv_autoencoder():
	pass

def test_conv_tied_autoencoder():
	pass

if __name__ == '__main__':
	print("\n\n\n\nStarting...\n\n\n\n")
	
	(x_train, _), (x_test, y_test) = mnist.load_data()
	x_train, x_test = flatten_input(x_train, x_test)
	test_autoencoder('mnist', x_train, x_test)


