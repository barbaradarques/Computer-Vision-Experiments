__author__ = "Barbara Darques"

import output2file as o2f
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import datasets
import time
import matplotlib.pyplot as plt

def load_accuracy_file(filename, separator):
	data = []
	with open(filename,'r') as input_file:
		for line in input_file.readlines():
			data.append(line.replace('\n','').split(separator))
	data = np.array(data)
	params = data[:, 0].astype("float64")  # parameter being changed along the test
	acc_vect = data[:, 1:].astype("float64") 
	return params, acc_vect

def test_linear_SVC_params(dataset_name, layer_name):
	names, values, classes = o2f.load_data('outputs/' + dataset_name +'/' + layer_name +'.txt', " ")
	values_train, values_test, classes_train, classes_test = train_test_split(values, classes, test_size=0.9, random_state=0)

	with open('svm_performance/' + dataset_name + '/accuracy-linear-' + layer_name + '.csv','w') as file:
		# file.write('# <cost>, <accuracy score vector>\n')
		for i in range(11):
			cost = 1 << i # penalty
			clf = svm.SVC(kernel = 'linear', C = cost)
			scores = cross_val_score(clf, values, classes, cv = 10)
			scores_str = ",".join(str(i) for i in scores)
			file.write(str(cost) + ',' + scores_str + '\n')
			# ====================================
		# 	plot_linear(clf, values, classes, False, i)
		# plt.show() # it's needed once 'show' is set to False in plot_linear <<<<

'''
def plot_linear(clf, X, Y, show, fignum): # !!! ignore for now !!!
	clf.fit(X, Y)

	# get the separating hyperplane
	w = clf.coef_[0]
	a = -w[0] / w[1]
	xx = np.linspace(-5, 5)
	yy = a * xx - (clf.intercept_[0]) / w[1]

	# plot the parallels to the separating hyperplane that pass through the
	# support vectors (margin away from hyperplane in direction
	# perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
	# 2-d.
	margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
	yy_down = yy - np.sqrt(1 + a ** 2) * margin
	yy_up = yy + np.sqrt(1 + a ** 2) * margin

	# plot the line, the points, and the nearest vectors to the plane
	plt.figure(fignum, figsize=(4, 3))
	plt.clf()
	plt.plot(xx, yy, 'k-')
	plt.plot(xx, yy_down, 'k--')
	plt.plot(xx, yy_up, 'k--')

	plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
				facecolors='none', zorder=10, edgecolors='k')
	plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
				edgecolors='k')

	plt.axis('tight')
	x_min = -4.8
	x_max = 4.2
	y_min = -6
	y_max = 6

	XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
	Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(XX.shape)
	plt.figure(fignum, figsize=(4, 3))
	plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)

	plt.xticks(())
	plt.yticks(())
	if(show):
		plt.show()
'''

def test_poly_SVC_params(dataset_name, layer_name, cost):
	names, values, classes = o2f.load_data('outputs/' + dataset_name + '/' + layer_name + '.txt', " ")
	values_train, values_test, classes_train, classes_test = train_test_split(values, classes, test_size=0.9, random_state=0)

	with open('svm_performance/' + dataset_name + '/accuracy-poly-' + layer_name + '.csv','w') as file:
		# file.write('# <degree>, <accuracy score vector>\n')
		for deg in range(2,8):
			clf = svm.SVC(kernel = 'poly', C = cost, degree = deg)
			scores = cross_val_score(clf, values, classes, cv = 10)
			scores_str = ",".join(str(i) for i in scores)
			file.write(str(deg) + ',' + scores_str + '\n')

def test_rbf_SVC_params(dataset_name, layer_name, cost):
	names, values, classes = o2f.load_data('outputs/' + dataset_name + '/' + layer_name + '.txt', " ")
	values_train, values_test, classes_train, classes_test = train_test_split(values, classes, test_size=0.9, random_state=0)

	with open('svm_performance/' + dataset_name + '/accuracy-rbf-' + layer_name + '.csv','w') as file:
		# file.write('# <gamma>, <accuracy score vector>\n')
		for i in range(9):
			g = (1<<i)/(4*values.shape[1])
			clf = svm.SVC(kernel = 'rbf', C = cost, gamma = g)
			scores = cross_val_score(clf, values, classes, cv = 10)
			scores_str = ",".join(str(i) for i in scores)
			file.write(str(g) + ',' + scores_str + '\n')


######################################################################################	
