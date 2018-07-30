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




# class TiedConv2D(_Conv):
#     @interfaces.legacy_conv2d_support
#     def __init__(self, filters,
#                  kernel_size,
#                  tied_to=None, tie_type='transpose',
#                  strides=(1, 1),
#                  padding='valid',
#                  data_format=None,
#                  dilation_rate=(1, 1),
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         super(TiedConv2D, self).__init__(
#             rank=2,
#             filters=filters,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding=padding,
#             data_format=data_format,
#             dilation_rate=dilation_rate,
#             activation=activation,
#             use_bias=use_bias,
#             kernel_initializer=kernel_initializer,
#             bias_initializer=bias_initializer,
#             kernel_regularizer=kernel_regularizer,
#             bias_regularizer=bias_regularizer,
#             activity_regularizer=activity_regularizer,
#             kernel_constraint=kernel_constraint,
#             bias_constraint=bias_constraint,
#             **kwargs)

#         self.tied_to = tied_to
#         self.tied_weights = tied_to.weights
#         self.tie_type = tie_type

#     def build(self, input_shape):
#         if self.data_format == 'channels_first':
#             channel_axis = 1
#         else:
#             channel_axis = -1
#         if input_shape[channel_axis] is None:
#             raise ValueError('The channel dimension of the inputs '
#                              'should be defined. Found `None`.')
#         input_dim = input_shape[channel_axis]
#         kernel_shape = self.kernel_size + (input_dim, self.filters)

#         # self.kernel = self.tied_to.weights

#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.filters,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#         # Set input spec.
#         self.input_spec = InputSpec(ndim=self.rank + 2,
#                                     axes={channel_axis: input_dim})
#         self.built = True


#     def call(self, inputs):
#         assert self.kernel_size[0] % 2 == 1, "Error: the kernel size is an even number"
#         assert self.kernel_size[1] % 2 == 1, "Error: the kernel size is an even number"

#         centerX = (self.kernel_size[0] - 1) // 2
#         centerY = (self.kernel_size[1] - 1) // 2

#         kernel_mask = np.ones(self.kernel_size + (1, 1))
#         kernel_mask[centerX, centerY] = 0
#         kernel_mask = K.variable(kernel_mask)

#         customKernel = self.kernel * kernel_mask

#         outputs = K.conv2d(
#                 inputs,
#                 self.tied_weights[0],
#                 strides=self.strides,
#                 padding=self.padding,
#                 data_format=self.data_format,
#                 dilation_rate=self.dilation_rate)

#         if self.use_bias:
#             outputs = K.bias_add(
#                 outputs,
#                 self.tied_weights[1],
#                 data_format=self.data_format)

#         if self.activation is not None:
#             return self.activation(outputs)

#         return outputs









