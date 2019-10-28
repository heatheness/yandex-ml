# -*- coding: utf-8 -*-

__author__ = 'nyash myash'

import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

data = pandas.read_csv("classification.csv", sep=',', header=None)
y_predict = data.values[1:,1].astype(np.float)
y_true = data.values[1:,0].astype(np.float)

tp = sum(np.logical_and((y_predict == np.ones(y_predict .shape)).astype(np.int),(y_true == np.ones(y_predict .shape))).astype(np.int))
fp = sum(np.logical_and((y_predict == np.ones(y_predict .shape)).astype(np.int),(y_true == np.zeros(y_predict .shape))).astype(np.int))
tn = sum(np.logical_and((y_predict == np.zeros(y_predict .shape)).astype(np.int),(y_true == np.zeros(y_predict .shape))).astype(np.int))
fn = sum(np.logical_and((y_predict == np.zeros(y_predict .shape)).astype(np.int),(y_true == np.ones(y_predict .shape))).astype(np.int))

print tp,fp,fn,tn

accuracy = accuracy_score(y_true,y_predict)
precision = precision_score(y_true,y_predict)
recall = recall_score(y_true,y_predict)
f1 = f1_score(y_true, y_predict)

print round(accuracy,2),round(precision,2),round(recall,2),round(f1,2)



