from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
print("Conv2D", end="= ")
print(encoded._keras_shape)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
print("MaxPooling2D", end="= ")
print(encoded._keras_shape)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
print("Conv2D", end="= ")
print(encoded._keras_shape)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
print("MaxPooling2D", end="= ")
print(encoded._keras_shape)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
print("Conv2D", end="= ")
print(encoded._keras_shape)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
print("MaxPooling2D", end="= ")
print(encoded._keras_shape)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
print("Conv2D", end="= ")
print(decoded._keras_shape)
decoded = UpSampling2D((2, 2))(decoded)
print("UpSampling2D", end="= ")
print(decoded._keras_shape)
decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)
print("Conv2D", end="= ")
print(decoded._keras_shape)
decoded = UpSampling2D((2, 2))(decoded)
print("UpSampling2D", end="= ")
print(decoded._keras_shape)
decoded = Conv2D(16, (3, 3), activation='relu')(decoded)
print("Conv2D", end="= ")
print(decoded._keras_shape)
decoded = UpSampling2D((2, 2))(decoded)
print("UpSampling2D", end="= ")
print(decoded._keras_shape)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
print("Conv2D", end="= ")
print(decoded._keras_shape)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')