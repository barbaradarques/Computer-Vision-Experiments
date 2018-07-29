import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
import tensorflow as tf

class TiedDenseLayer(Layer):
	def __init__(self, output_dim, tied_to, tie_type = 'transpose', activation=None, **kwargs):
		super(TiedDenseLayer, self).__init__(**kwargs)
		self.output_dim = output_dim
		self.activation = activations.get(activation)
		self.tied_to = tied_to
		self.tied_weights = self.tied_to.weights
		self.tie_type = tie_type
		

	def build(self, input_shape):
		super(TiedDenseLayer, self).build(input_shape)


	def call(self, x):
		# <layer>.weights[0] = actual weights
		# <layer>.weights[1] = bias
		print(self.tied_to.get_weights()[0])

		if self.tie_type == 'transpose':
				output = K.dot(x - self.tied_weights[1], K.transpose(self.tied_weights[0]))
		elif self.tie_type == 'inverse':
				output = K.dot(x - self.tied_weights[1], tf.convert_to_tensor(np.linalg.pinv(self.tied_to.get_weights()[0])))
		
		if self.activation is not None:
				output = self.activation(output)

		return output


	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)


	def get_config(self):
		config = {
				  'activation': activations.serialize(self.activation),
				  'tie_type': self.tie_type
				}
		base_config = super(TiedDenseLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


##############################

def main():
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	# prepare input data
	(x_train, _), (x_test, y_test) = mnist.load_data()

	# normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	print(x_train.shape)
	print(x_test.shape)

# if __name__ = '__main__':
