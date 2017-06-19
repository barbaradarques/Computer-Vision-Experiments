from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import os

cnn = VGG16(weights='imagenet')
cnn.summary() # full description of the architecture
1/0
# for i, layer in enumerate(cnn.layers): # prints the layers names (so you can know how to refer to each one in "get_layer")
#    print(i, layer.name)

def batch_preprocessing(dir_name): #'Produce_1400'
	subdirs = next(os.walk(dir_name))[1] # returns all the subdirectories inside Produce_1400
	all_imgs = []
	all_imgs_names = []
	all_imgs_classes = []
	for subdir in subdirs:
		imgs = [subdir+'/'+img for img in os.listdir('./'+dir_name+'/'+subdir) if img.endswith('.jpg')]
		imgs_classes =np.empty(len(imgs))
		print(len(imgs_classes))
		imgs_classes.fill(subdir) # as the classes are separated by subdirectories, class name = subdir name
		all_imgs_classes.extend(imgs_classes)
		all_imgs_names.extend(imgs)
	print(all_imgs_names)
	print(all_imgs_classes)


def get_layer_output(cnn, layers_names, img_path):
	layers = [cnn.get_layer(layer_name).output for layer_name in layers_names]
	cnn = Model(inputs=cnn.input, outputs=layers)
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return cnn.predict(x)

fc1 = get_layer_output(cnn, 'fc1', 'elephant.jpg')
names = np.array(['elephant.jpg', 'blablabla.jpg'])
# print(fc1)
# print(fc1.shape)


# ab = np.zeros(names.size, dtype=[('var1', 'U6'), ('var2', float)])
# ab['var1'] = names
# ab['var2'] = fc1[:2]

# np.savetxt('fc1.txt', names, delimiter= ",", fmt="%s")
# np.savetxt('fc1.txt', ab, fmt="%s %f")
fc = [fc1[:2], fc1[:2]]
final = np.r_['1,2,0', names, fc]
final2 = np.r_['1,2,0', final, names]
final2 = np.asarray(final2)
print(final2)
# np.savetxt('fc1.txt', final2)
# print(final2.dtype)



# labels = ['label1', 'label2', 'label3']
# values = [[0.1, 0.4, 0.5],
# 		  [0.1, 0.2, 0.1],
# 		  [0.5, 0.6, 1.0]]
# descriptions = ['desc1', 'desc2', 'desc3']
# concat1 = np.r_['1,2,0', labels, values]
# concat2 = np.r_['1,2,0', concat1, descriptions]
# print(concat2)

# file = open('output.txt', mode='w')
# concat2.tofile(file, sep='/')
# np.savetxt('fc1.txt', concat2, fmt = "%s")
# recovered = np.loadtxt('fc1.txt', converters={0: lambda x: str(x, 'utf-8')}, dtype=str)
print('recovered:')

# file = open('output.txt', mode='r')
# recovered = np.fromfile(file, sep='/', dtype=None)
def save_data(filename, imgs_names, output_values, imgs_classes):
	concat1 = np.r_['1,2,0', imgs_names, output_values]
	concat2 = np.r_['1,2,0', concat1, imgs_classes]
	with open(filename, "ab") as output_file: # append to the end of the file
		np.savetxt(output_file, concat2, fmt = "%s")

def load_data(filename, separator):
	data = []
	with open(filename,'r') as input_file:
		for line in input_file.readlines():
			data.append(line.replace('\n','').split(separator))
	return data

recovered = load_data('fc1.txt', ' ') 

print(recovered)

# file.close()
"""
[['label1' '0.1' '0.4' '0.5' 'desc1']
 ['label2' '0.1' '0.2' '0.1' 'desc2']
 ['label3' '0.5' '0.6' '1.0' 'desc3']]
"""
