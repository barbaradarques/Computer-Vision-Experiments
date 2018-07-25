
from svm_tests import load_accuracy_file
import numpy as np
import matplotlib as mpl 
from matplotlib.ticker import FormatStrFormatter
from decimal import Decimal

mpl.use('agg') # agg backend is used to create plot as a .png file
import matplotlib.pyplot as plt 

def plot_svm_performance(dataset_name, layer_name, **kwargs):
	linear_params, linear_data = load_accuracy_file('svm_performance/'+ dataset_name +'/accuracy-linear-'+ layer_name +'.csv', ',')
	linear_params = [int(param) for param in linear_params]
	linear_data = linear_data.tolist() # otherwise it reads the wrong axes | another solution: linear_data = np.swapaxes(linear_data, 0, 1)

	poly_params, poly_data = load_accuracy_file('svm_performance/'+ dataset_name +'/accuracy-poly-'+ layer_name +'.csv', ',')
	poly_params = [int(param) for param in poly_params]
	poly_data = poly_data.tolist() # otherwise it reads the wrong axes | another solution: poly_data = np.swapaxes(poly_data, 0, 1)

	rbf_params, rbf_data = load_accuracy_file('svm_performance/'+ dataset_name +'/accuracy-rbf-'+ layer_name +'.csv', ',')
	rbf_params = ['%.0e' % Decimal(param) for param in rbf_params]
	rbf_data = rbf_data.tolist() # otherwise it reads the wrong axes | another solution: rbf_data = np.swapaxes(rbf_data, 0, 1)

	fig = plt.figure(figsize=(15.0, 10.0)) # 1,
	# fig.suptitle(dataset_name + ' (' + layer_name + ') - SVM Results\n\n', fontsize=16)
	plt.title(dataset_name + ' (' + layer_name + ') - SVM Results\n\n', fontsize=16)
	plt.axis('off')

	sp = fig.add_subplot(221)
	sp.set_ylabel("Accuracy")
	sp.set_xlabel("Cost")
	bp = sp.boxplot(linear_data, labels = linear_params)
	for flier in bp['fliers']:
	    flier.set(marker='+')
	sp.title.set_text('Using Linear Kernel')
	sp.set_ylim([0, 1])


	sp = fig.add_subplot(222)
	sp.set_ylabel("Accuracy")
	sp.set_xlabel("Polynomial Degree")
	bp = sp.boxplot(poly_data, labels = poly_params) # alternative to 'labels1 param => sp.set_xticklabels(params)
	for flier in bp['fliers']:
	    flier.set(marker='+')
	sp.title.set_text('Using Polynomial Kernel')
	sp.set_ylim([0, 1])

	sp = fig.add_subplot(223)
	sp.set_ylabel("Accuracy")
	sp.set_xlabel("Gamma")
	bp = sp.boxplot(rbf_data, labels = rbf_params)
	for flier in bp['fliers']:
	    flier.set(marker='+')
	sp.title.set_text('Using RBF Kernel')
	# sp.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
	fig.tight_layout()
	sp.set_ylim([0, 1])
	
	if 'show' in kwargs and kwargs['show'] == True:
		plt.show()

	fig.savefig('boxplots/'+dataset_name+'/' + layer_name + '-boxplot.png', dpi=100, bbox_inches='tight')
	del fig # <<<
####################################################

def plot_svm_performance_per_dataset(dataset_name, layers_names, **kwargs):
	fig = plt.figure(figsize=(15.0, 10.0)) # 1,
	# fig.suptitle(dataset_name + ' (' + layer_name + ') - SVM Results\n\n', fontsize=16)
	plt.title(dataset_name + ' - SVM Results\n\n', fontsize=16)
	plt.axis('off')

	for i, layer_name in enumerate(layers_names):
		sp = fig.add_subplot(5, 3, 3 * i + 1)
		sp.set_ylabel("Accuracy")
		sp.set_xlabel("Cost")
		linear_params, linear_data = load_accuracy_file('svm_performance/'+ dataset_name +'/accuracy-linear-'+ layer_name +'.csv', ',')
		linear_data = linear_data.tolist() # otherwise it reads the wrong axes | another solution: linear_data = np.swapaxes(linear_data, 0, 1)
		linear_params = [int(param) for param in linear_params]
		bp = sp.boxplot(linear_data, labels = linear_params, showfliers=False)
		sp.title.set_text(layer_name + ' - Using Linear Kernel')
		sp.set_ylim([0, 1])


		sp = fig.add_subplot(5, 3, 3 * i + 2)
		sp.set_ylabel("Accuracy")
		sp.set_xlabel("Polynomial Degree")
		poly_params, poly_data = load_accuracy_file('svm_performance/'+ dataset_name +'/accuracy-poly-'+ layer_name +'.csv', ',')
		poly_data = poly_data.tolist() # otherwise it reads the wrong axes | another solution: poly_data = np.swapaxes(poly_data, 0, 1)
		poly_params = [int(param) for param in poly_params]
		bp = sp.boxplot(poly_data, labels = poly_params, showfliers=False) # alternative to 'labels1' param => sp.set_xticklabels(params)
		sp.title.set_text(layer_name + ' - Using Polynomial Kernel')
		sp.set_ylim([0, 1])

		sp = fig.add_subplot(5, 3, 3 * i + 3)
		sp.set_ylabel("Accuracy")
		sp.set_xlabel("Gamma")
		rbf_params, rbf_data = load_accuracy_file('svm_performance/'+ dataset_name +'/accuracy-rbf-'+ layer_name +'.csv', ',')
		rbf_data = rbf_data.tolist() # otherwise it reads the wrong axes | another solution: rbf_data = np.swapaxes(rbf_data, 0, 1)
		rbf_params = ['%.0e' % Decimal(param) for param in rbf_params]
		bp = sp.boxplot(rbf_data, labels = rbf_params, showfliers=False)
		sp.title.set_text(layer_name + ' - Using RBF Kernel')
		fig.tight_layout()
		sp.set_ylim([0, 1])
	
	if 'show' in kwargs and kwargs['show'] == True:
		plt.show()

	fig.savefig('boxplots/'+dataset_name+'/' + 'summary-boxplot.png', dpi=100, bbox_inches='tight')
	del fig # <<<


