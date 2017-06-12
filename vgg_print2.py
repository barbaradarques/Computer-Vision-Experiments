from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np

cnn = VGG16(weights='imagenet')
# cnn.summary() # full descritption
# for i, layer in enumerate(cnn.layers): # prints the layers names (so you can know how to refer to each one in "get_layer")
#    print(i, layer.name)
def get_layer_output(cnn, layer_name, img_path):
	cnn = Model(inputs=cnn.input, outputs=cnn.get_layer('fc1').output)
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return cnn.predict(x)[0]

fc1 = get_layer_output(cnn, 'fc1', 'elephant.jpg')
names = np.array(['elephant.jpg', 'blablabla.jpg'])
print(fc1)
print(fc1.shape)


# ab = np.zeros(names.size, dtype=[('var1', 'U6'), ('var2', float)])
# ab['var1'] = names
# ab['var2'] = fc1[:2]

# np.savetxt('fc1.txt', names, delimiter= ",", fmt="%s")
# np.savetxt('fc1.txt', ab, fmt="%s %f")
fc = [fc1[:2], fc1[:2]]
final = np.r_['1,2,0', names, fc]
final2 = np.r_['1,2,0', final, names]
print(final2)

