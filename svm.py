__author__ = "Barbara Darques"

import output2file as o2f
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import datasets
import time

def test_linear_SVC_params(dataset_name, layer_output_file):
	names, values, classes = o2f.load_data(layer_output_file, " ")
	values_train, values_test, classes_train, classes_test = train_test_split(values, classes, test_size=0.9, random_state=0)

	with open('svm_performance/'+dataset_name+'/accuracy-linear-'+layer_output_file[:-4]+'.csv','w') as file:
		file.write('# <cost>, <accuracy score vector>\n')
		for i in range(11):
			cost = 1 << i # penalty = ??? <<<<<<<<
			clf = svm.SVC(kernel='linear', C=cost)
			scores = cross_val_score(clf, values, classes, cv=5)
			scores_str = ",".join(str(i) for i in scores)
			file.write(str(cost)+','+scores_str+'\n')

def test_poly_SVC_params(dataset_name, layer_output_file, cost):
	names, values, classes = o2f.load_data(layer_output_file, " ")
	values_train, values_test, classes_train, classes_test = train_test_split(values, classes, test_size=0.9, random_state=0)

	with open('svm_performance/'+dataset_name+'/accuracy-poly-'+layer_output_file[:-4]+'.csv','w') as file:
		file.write('# <degree>, <accuracy score vector>\n')
		for deg in range(2,8):
			clf = svm.SVC(kernel='poly', C=cost, degree = deg)
			scores = cross_val_score(clf, values, classes, cv=5)
			scores_str = ",".join(str(i) for i in scores)
			file.write(str(deg)+','+scores_str+'\n')

def test_rbf_SVC_params(dataset_name, layer_output_file, cost):
	names, values, classes = o2f.load_data(layer_output_file, " ")
	values_train, values_test, classes_train, classes_test = train_test_split(values, classes, test_size=0.9, random_state=0)

	with open('svm_performance/'+dataset_name+'/accuracy-rbf-'+layer_output_file[:-4]+'.csv','w') as file:
		file.write('# <gamma>, <accuracy score vector>\n')
		for i in range(9):
			g = (1<<i)/(4*values.shape[1])
			clf = svm.SVC(kernel='rbf', C=cost, gamma = g)
			scores = cross_val_score(clf, values, classes, cv=5)
			scores_str = ",".join(str(i) for i in scores)
			file.write(str(g)+','+scores_str+'\n')


######################################################################################	
start_time = time.time()

layer_output_file = "produce-fc1.txt"
# test_linear_SVC_params('Produce_1400', layer_output_file)
test_rbf_SVC_params('Produce_1400', layer_output_file, 1)

print("\n\n\n\nExecution time: %s seconds.\n\n\n\n" % (time.time() - start_time))
