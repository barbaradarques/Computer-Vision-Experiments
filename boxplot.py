
from svm import load_accuracy_file
import numpy as np
import matplotlib as mpl 
from matplotlib.ticker import FormatStrFormatter

mpl.use('agg') # agg backend is used to create plot as a .png file
import matplotlib.pyplot as plt 

def plot_svm_performance(dataset_name, layer_name):
	linear_params, linear_data = load_accuracy_file('svm_performance/'+ dataset_name +'/accuracy-linear-'+ layer_name +'.csv', ',')
	linear_data = linear_data.tolist() # otherwise it reads the wrong axes | another solution: linear_data = np.swapaxes(linear_data, 0, 1)

	poly_params, poly_data = load_accuracy_file('svm_performance/'+ dataset_name +'/accuracy-poly-'+ layer_name +'.csv', ',')
	poly_data = poly_data.tolist() # otherwise it reads the wrong axes | another solution: poly_data = np.swapaxes(poly_data, 0, 1)

	rbf_params, rbf_data = load_accuracy_file('svm_performance/'+ dataset_name +'/accuracy-rbf-'+ layer_name +'.csv', ',')
	rbf_data = rbf_data.tolist() # otherwise it reads the wrong axes | another solution: rbf_data = np.swapaxes(rbf_data, 0, 1)

	fig = plt.figure(1)
	fig.suptitle(dataset_name + ' (' + layer_name + ') - SVM Results', fontsize=12)


	sp = fig.add_subplot(221)
	sp.set_ylabel("Accuracy")
	sp.set_xlabel("Cost")
	bp = sp.boxplot(linear_data, labels = linear_params)
	for flier in bp['fliers']:
	    flier.set(marker='+')
	sp.title.set_text('Using Linear Kernel')


	sp = fig.add_subplot(222)
	sp.set_ylabel("Accuracy")
	sp.set_xlabel("Polynomial Degree")
	bp = sp.boxplot(poly_data, labels = poly_params) # alternative to 'labels1 param => sp.set_xticklabels(params)
	for flier in bp['fliers']:
	    flier.set(marker='+')
	sp.title.set_text('Using Polynomial Kernel')

	sp = fig.add_subplot(223)
	sp.set_ylabel("Accuracy")
	sp.set_xlabel("Gamma")
	bp = sp.boxplot(rbf_data, labels = rbf_params)
	for flier in bp['fliers']:
	    flier.set(marker='+')
	sp.title.set_text('Using RBF Kernel')
	# sp.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
	fig.tight_layout()
	plt.show()

	fig.savefig('boxplots/'+dataset_name+'/' + layer_name + '-boxplot.png', bbox_inches='tight')

####################################################


