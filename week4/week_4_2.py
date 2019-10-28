# -*- coding: utf-8 -*-

__author__ = 'nyash myash'

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import bisect

data = pd.read_csv('close_prices.csv')
X = data.values[:,1:]
print X.shape

dow_jones = pd.read_csv('djia_index.csv').values[:,1]
pca = PCA(n_components=10)
pca.fit(X)
print pca.components_.shape
res = pca.transform(X)[:,0]
print pca.transform(X).shape

print bisect.bisect_left(np.cumsum(pca.explained_variance_ratio_), 0.9)
print np.round(np.corrcoef(res,dow_jones), 2)

idx = np.argmax(pca.components_[0])
# print pca.components_[0]

print data.columns[1:][idx]

# print X_2



# print pca.explained_variance_ratio_
