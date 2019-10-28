# -*- coding: utf-8 -*-

__author__ = 'nyash myash'

import pandas
import math
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score


data = pandas.read_csv("data-logistic.csv", sep=',', header=None)
x = data.values[:, 1:]
y = data.values[:, 0]

sample_size = x.shape[0]

def gradient_descent(x,y,initial=np.array([0.0,0.0]), k=0.1, l=sample_size, eps=1e-5):
    w = initial
    # w[0] = w_prev[0] + k*(1.0/l)*sum(y[0]*x[i,0]*(1-(1/(1+math.exp(-y[i]*(w*x[i,:]))))))

    for item in xrange(10000):
        w_temp = np.copy(w)
        w[0] = w_temp[0] + k*(1.0/l)*sum(y*x[:,0]*(1-(1.0/(1+np.exp(-y*(np.dot(x,w_temp)))))))
        w[1] = w_temp[1] + k*(1.0/l)*sum(y*x[:,1]*(1-(1.0/(1+np.exp(-y*(np.dot(x,w_temp)))))))
        if distance.euclidean(w, w_temp) < eps:
            return w
    return w

def gradient_descent_regularized(x,y,initial=np.array([0.0,0.0]), k=0.1, l=sample_size, c=10.0, eps=1e-5):
    w = initial
    # w[0] = w_prev[0] + k*(1.0/l)*sum(y[0]*x[i,0]*(1-(1/(1+math.exp(-y[i]*(w*x[i,:]))))))

    for item in xrange(10000):
        w_temp = np.copy(w)
        w[0] = w_temp[0] + k*(1.0/l)*sum(y*x[:,0]*(1-(1.0/(1+np.exp(-y*(np.dot(x,w_temp))))))) - k*c*w[0]
        w[1] = w_temp[1] + k*(1.0/l)*sum(y*x[:,1]*(1-(1.0/(1+np.exp(-y*(np.dot(x,w_temp))))))) - k*c*w[1]
        if distance.euclidean(w, w_temp) <= eps:
            return w
    return w

def sigmoid(x,w):
    return 1.0/(1.0 + np.exp(-(np.dot(x,w))))

print gradient_descent(x,y)
print gradient_descent_regularized(x,y)

w_unregularized = gradient_descent(x,y)
w_regularized = gradient_descent_regularized(x,y)


y_predict_unregularized = sigmoid(x,w_unregularized)
y_predict_regularized = sigmoid(x,w_regularized)

print roc_auc_score(y, y_predict_unregularized)
print roc_auc_score(y,y_predict_regularized)