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











