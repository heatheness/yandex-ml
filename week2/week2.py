# -*- coding: utf-8 -*-

__author__ = 'nyash myash'

import pandas
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pandas.read_csv('wine.csv', sep=',', header=None)


x = data.values[:,1:]
y = data.values[:,0]
x = preprocessing.scale(x, axis=0, with_mean=True, with_std=True, copy=True)

m = x.shape[0]

kf = KFold(m, n_folds=5, shuffle=True, random_state=42)
print kf

max_accuracy = 0
best_k = 1

for k in xrange(1, 51):
    acc = []
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train, y_train)
        score = neigh.score(x_test, y_test) # a smart way to calculate accuracy
        acc.append(score)
        # a = neigh.predict(x_test)
        # res = a == y_test
        # res = res.astype(float)
        # # print res
        # accuracy = sum(res) / res.shape[0]
        # acc.append(accuracy)
    mean_acc = np.mean(acc)
    if mean_acc > max_accuracy:
        max_accuracy = mean_acc
        best_k = k

print max_accuracy
print best_k

