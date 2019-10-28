# -*- coding: utf-8 -*-

__author__ = 'nyash myash'



# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.svm import SVC


data = pandas.read_csv("svm-data.csv", sep=',', header=None)
clf = SVC(C=10000, kernel='linear', random_state=241)
clf.fit(data.values[:, 1:], data.values[:, 0])
print clf.support_



class_0 = data[data[0] == 0]
class_1 = data[data[0] == 1]

plt.plot(class_0[1], class_0[2], 'bo', c='red')
plt.plot(class_1[1], class_1[2], 'bo', c='blue')
plt.xlabel('x')
plt.ylabel('y')

# —--- Shameless copy-paste below —--- #

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 1)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')


# plt.draw() - for jupiter
plt.show()

