# -*- coding: utf-8 -*-

__author__ = 'nyash myash'

import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold

import heapq


newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

x = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(x,y)
# vectorizer.transform(y)



grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)


# clf = SVC(kernel='linear', random_state=241)
# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# gs.fit(tfidf, y)
#
#
# for a in gs.grid_scores_:
#     print a.mean_validation_score
#     print a.parameters

# print gs.best_score_
# print gs.best_params_
# print gs.best_params_['C']


clf = SVC(C=1.0, kernel='linear', random_state=241)
clf.fit(tfidf, y)

words = vectorizer.get_feature_names()
# print clf.coef_
cfs = clf.coef_.toarray()[0]
# print cfs

cfs = map(lambda z: abs(z), cfs)
cfs_top = heapq.nlargest(10, cfs)
words_top = []

for item in cfs_top:
    for i in xrange(len(cfs)):
        if cfs[i] == item:
            words_top.append(words[i])

words_top.sort()
print words_top