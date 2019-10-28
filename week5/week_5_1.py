# -*- coding: utf-8 -*-

__author__ = 'nyash myash'


import pandas
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


data_raw = pandas.read_csv("gbm-data.csv", skipinitialspace=True)
data = data_raw.values
x = data[:, 1:]
y = data[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=241)
learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
res = {}

for item in learning_rate:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=item)
    clf.fit(x_train, y_train)
    train_pred = clf.staged_predict_proba(x_train)
    test_pred = clf.staged_predict_proba(x_test)
    train_loss = [log_loss(y_train, pr) for pr in train_pred]
    test_loss = [log_loss(y_test,pr) for pr in test_pred]
    iter_min = np.argmin(test_loss)
    res[item] = (iter_min, test_loss[iter_min])
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.title('log-loss{}'.format(item))
    plt.savefig('learning rate{}.png'.format(str(item)))


print res


# clf = RandomForestClassifier(n_estimators=36, random_state=241)
# clf.fit(x_train, y_train)
# test_pred = clf.predict_proba(x_test)
# loss = log_loss(y_test, test_pred)
# print loss


