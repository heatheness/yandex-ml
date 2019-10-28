# -*- coding: utf-8 -*-

__author__ = 'nyash myash'

import pandas
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler



test = pandas.read_csv('perceptron-test.csv', sep=',', header=None)
train = pandas.read_csv('perceptron-train.csv', sep=',', header=None)


clf = Perceptron(random_state=241)
clf.fit(train.values[:,1:], train.values[:, 0])


predictions = clf.predict(test.values[:, 1:])

print predictions

accuracy = metrics.accuracy_score(test.values[:, 0], predictions)
print accuracy

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train.values[:, 1:])
test_scaled = scaler.transform(test.values[:, 1:])
clf.fit(train_scaled, train.values[:, 0])
predictions_s = clf.predict(test_scaled)
accuracy_s = metrics.accuracy_score(test.values[:, 0], predictions_s)
print accuracy_s

print accuracy_s - accuracy

