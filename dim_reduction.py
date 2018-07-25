import matplotlib
matplotlib.use('Agg')
import time
import numpy as np
import output2file as o2f
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model

def tsne_reduction(dataset_name, layer_name):
	print('Loading CNN data...')
	names, values, classes = o2f.load_data('outputs/' + dataset_name +'/' + layer_name +'.txt', " ")
	print('Calculating T-SNE...')
	tsne_data = TSNE(n_components=2, init='pca').fit_transform(values)
	print('Saving T-SNE results...')
	with open('t-sne_results/' + dataset_name + '/' + layer_name + '.csv','w') as file:
		for point in tsne_data:
			file.write(str(point[0]) + ',' + str(point[1]) + '\n')
	return tsne_data


def load_2d_data(dataset_name, layer_name):
	data = []
	with open('t-sne_results/' + dataset_name + '/' + layer_name + '.csv','r') as file:
		for line in file.readlines():
			data.append(line.replace('\n','').split(','))
	return np.array(data).astype('float64')

def plot_tsne_results(dataset_name, layer_name):
	print('Plotting T-SNE results...')
	tsne_data = load_2d_data(dataset_name, layer_name)
	print('shape = ', str(tsne_data.shape))
	print('Loading classes... => from:'+ 'outputs/' + dataset_name +'/' + layer_name +'.txt')
	
	plt.figure()
	_ , _ , classes = o2f.load_data('outputs/' + dataset_name +'/' + layer_name +'.txt', " ")

	num_classes = len(set(classes))
	colors = []
	color_step = 1/num_classes
	last_color = 0
	for i in range(num_classes):
		last_color += color_step
		colors.append(str(last_color))

	color_class_map = dict(zip(colors, classes))
	color_class_map2 = dict(zip(classes, colors))
	tsne_data = [ [data[0], data[1], color_class_map2[classes[i]]] for i, data in enumerate(tsne_data)] 
	for color in colors:
		sub = [data for data in tsne_data if data[2]==color]
		plt.scatter(sub[0], sub[1], color=color, alpha=.8, lw=1, label=color_class_map[color])
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('T-SNE representation of ' + dataset_name + ' - ' + layer_name)

	# plt.show()
	fig.savefig('t-sne_graphs/'+dataset_name+'/' + layer_name + '_t-sne.png', dpi=100, bbox_inches='tight')


def main1():
	cnn = VGG16(weights='imagenet')
	# layers_names = [layer.name for layer in cnn.layers[-8:-1] if (layer.name[-4:] != 'pool') and (layer.name != 'flatten')] # !!! using just block5_conv1 for now !!!
	layers_names = ['block5_conv1','block5_conv2','block5_conv3','fc1','fc2']	
	datasets_names = ['17flowers', 'coil-20', 'corel-1000', 'tropical_fruits1400']
	for dataset_name in datasets_names:
		for layer_name in layers_names:
			print(dataset_name + ' - ' + layer_name)
			tsne_data = tsne_reduction(dataset_name, layer_name)
			print(tsne_data.shape)


def main2():
	cnn = VGG16(weights='imagenet')
	# layers_names = [layer.name for layer in cnn.layers[-8:-7] if (layer.name[-4:] != 'pool') and (layer.name != 'flatten')] # !!! using just block5_conv1 for now !!

	datasets_names = ['17flowers', 'coil-20', 'corel-1000', 'tropical_fruits1400']
	layers_names = ['block5_conv1']
	for dataset_name in datasets_names:
		for layer_name in layers_names:
			print(dataset_name + ' - ' + layer_name)
			plot_tsne_results(dataset_name, layer_name)


if __name__ == '__main__':
	start_time = time.time()
	main2()
	print("\n\n\n\nExecution time: %s seconds.\n\n\n\n" % (time.time() - start_time))
	
