import numpy as np
import output2file as o2f
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
	print('Loading classes...')
	_ , _ , classes = o2f.load_data('outputs/' + dataset_name +'/' + layer_name +'.txt', " ")

	num_classes = len(set(classes))
	colors = []
	color_step = 1/num_classes
	last_color = 0
	for i in num_classes:
		last_color += color_step
		colors.append(str(last_color))

	plt.figure()

	for color, i, target_name in zip(colors, [0, 1, 2], target_names):
	    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=1,
	                label=classes[i])
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('T-SNE representation of ' + dataset_name + ' - ' + layer_name)

	# plt.show()
	fig.savefig('t-sne_graphs/'+dataset_name+'/' + layer_name + '_t-sne.png', dpi=100, bbox_inches='tight')

def main():
	cnn = VGG16(weights='imagenet')
	layers_names = [layer.name for layer in cnn.layers[-8] if (layer.name[-4:] != 'pool') and (layer.name != 'flatten')] # !!! using just block5_conv1 for now !!!
	datasets_names = ['17flowers', 'coil-20', 'corel-1000', 'tropical_fruits1400']
	for dataset_name in datasets_names:
		for layer_name in layers_names:
			print(dataset_name + ' - ' + layer_name)
			tsne_data = tsne_reduction(dataset_name, layer_name)
			print(tsne_data.shape)

if __name__ == '__main__':
	start_time = time.time()
	main()
	print("\n\n\n\nExecution time: %s seconds.\n\n\n\n" % (time.time() - start_time))
	