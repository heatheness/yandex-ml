# -*- coding: utf-8 -*-

__author__ = 'nyash myash'


import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score


data = pandas.read_csv("abalone.csv", skipinitialspace=True)
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

x = data.values[:, :8].astype(np.float)
y = data.values[:, 8].astype(np.float)
kf = KFold(len(x), n_folds=5, shuffle=True, random_state=1)


# for n_trees in xrange(1,51):
#     clf = RandomForestRegressor(n_estimators=n_trees, random_state=1)
#     res = []
#     for train_index, test_index in kf:
#         x_train, x_test = x[train_index], x[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf.fit(x_train, y_train)
#         pr = clf.predict(x_test)
#         score = r2_score(y_test,  pr)
#         res.append(score)
#     print n_trees, np.mean(res)
#     if np.mean(res) > 0.52:
#         print n_trees
#         n = n_trees



for n_trees in xrange(1,51):
    clf = RandomForestRegressor(n_estimators=n_trees, random_state=1)
    cvs = cross_val_score(clf, x, y, scoring='r2', cv=kf)
    print np.mean(cvs)
    print n_trees
    if np.mean(cvs) > 0.52:
        res = n_trees
        print n_trees
        break
#
with open('5.txt', 'w') as f:
    f.write(str(res))
