# -*- coding: utf-8 -*-

__author__ = 'nyash myash'

import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score

boston = load_boston()

x = boston.data
x = preprocessing.scale(x, axis=0, with_mean=True, with_std=True, copy=True)
m = x.shape[0]
y = boston.target
greed = np.linspace(1.0, 10.0, num=200)
best_p = 1

mse = {}


kf = KFold(m, n_folds=5, shuffle=True, random_state=42)

for power in greed:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p=power, metric='minkowski')
    cvs = cross_val_score(neigh, x, y, cv=kf)
    # print cvs
    mse[power] = max(cvs)

print mse
res = max(mse.iteritems(), key=lambda x:x[1])
print res




    # for train_index, test_index in kf:
    #     x_train, x_test = x[train_index], x[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     neigh.fit(x_train, y_train)
    #     cvs = cross_val_score(neigh, x_test, y=y_test, scoring='mean_squared_error', cv=kf)
    #
    #
    #     # y_pred = neigh.predict(x_test)
    #     # error = mean_squared_error(y_test, y_pred)
    #     # mse.append(error)
    #
    # # mean_acc = np.mean(acc)
    # if mean_acc > max_accuracy:
    #     max_accuracy = mean_acc

