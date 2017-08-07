import output2file as o2f
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import datasets

layer_output_file = "produce-fc1.txt"
names, values, classes = o2f.load_data(layer_output_file, " ")

# values_train, values_test, classes_train, classes_test = train_test_split(values, classes, test_size=0.9, random_state=0)


kernels = ['linear', 'poly', 'sigmoid', 'rbf']
with open('accuracy.csv','w') as file:
	file.write('dataset+layer, kernel, accuracy, standard deviation\n')
	for krn in kernels:
		clf = svm.SVC(kernel=krn) # 'linear', 'poly', 'sigmoid' or 'rbf'
		# clf = clf.fit(values_train, classes_train)
		# print(clf.score(values_test, classes_test))
		# print(clf.predict(values_test[:10]))
		# print(classes_test[:10])
		scores = cross_val_score(clf, values, classes, cv=5)
		file.write(layer_output_file + ',' + krn + ',' + str(scores.mean()) + ',' + str(scores.std()) +'\n')

# print(scores)
# print("Average Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# predicted = cross_val_predict(clf, values, classes, cv=10)
# print(predicted[:5])

# TODO:
# X flatten output 
# produce -> all layers
# svm -> clf for each kernel