# -*- coding: utf-8 -*-

__author__ = 'nyash myash'


import pandas
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

data = pandas.read_csv("scores.csv", sep=",", header=None)
# print data


y_true = data.values[1:,0].astype(np.float)
logreg = data.values[1:,1].astype(np.float)
svm = data.values[1:,2].astype(np.float)
knn = data.values[1:,3].astype(np.float)
tree = data.values[1:,4].astype(np.float)

scores = np.array([logreg,svm,knn,tree])


res = map(lambda x:roc_auc_score(y_true,x), scores)
print res.index(max(res))

max_precision = 0.0
score_num = 0


for num,score in enumerate(scores):
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    prc = []
    for i in xrange(len(recall)):
        if recall[i] >= 0.7:
            prc.append(precision[i])
    print max(prc)
    if max(prc) > max_precision:
        max_precision = max(prc)
        score_num = num

print num

# another way

import matplotlib.pyplot as plt
# Assignment 4 - Quality metrics

# %matplotlib inline

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

data = pd.read_csv('classification.csv', header=0)
classes_true = data.true
classes_predicted = data.pred

tp = 0
fp = 0
fn = 0
tn = 0
for i, class_true in enumerate(classes_true):
    class_predicted = classes_predicted[i]
    if class_predicted == 1 and class_true == 1:
        tp += 1
    elif class_predicted == 1 and class_true == 0:
        fp += 1
    elif class_predicted == 0 and class_true == 1:
        fn += 1
    else:
        tn += 1

# tp = sum(np.logical_and((y_predict == np.ones(y_predict .shape)).astype(np.int),(y_true == np.ones(y_predict .shape))).astype(np.int))
# fp = sum(np.logical_and((y_predict == np.ones(y_predict .shape)).astype(np.int),(y_true == np.zeros(y_predict .shape))).astype(np.int))
# tn = sum(np.logical_and((y_predict == np.zeros(y_predict .shape)).astype(np.int),(y_true == np.zeros(y_predict .shape))).astype(np.int))
# fn = sum(np.logical_and((y_predict == np.zeros(y_predict .shape)).astype(np.int),(y_true == np.ones(y_predict .shape))).astype(np.int))

print "TP: ", tp, " FP: ", fp
print "FN: ", fn, " TN: ", tn

print "\nAccuracy: ", metrics.accuracy_score(classes_true, classes_predicted)
print "Precision: ", metrics.precision_score(classes_true, classes_predicted)
print "Recall: ", metrics.recall_score(classes_true, classes_predicted)
print "F1: ", metrics.f1_score(classes_true, classes_predicted)

data = pd.read_csv('scores.csv', header=0)
classes_true = data.true
classes_logreg = data.score_logreg
classes_svm = data.score_svm
classes_knn = data.score_knn
classes_tree = data.score_tree
print "\nAUC-ROC logreg: ", metrics.roc_auc_score(classes_true, classes_logreg)
print "AUC-ROC svm: ", metrics.roc_auc_score(classes_true, classes_svm)
print "AUC-ROC knn: ", metrics.roc_auc_score(classes_true, classes_knn)
print "AUC-ROC tree: ", metrics.roc_auc_score(classes_true, classes_tree)


precision_logreg, recall_logreg, _ = metrics.precision_recall_curve(classes_true, classes_logreg)
precision_svm, recall_svm, _ = metrics.precision_recall_curve(classes_true, classes_svm)
precision_knn, recall_knn, _ = metrics.precision_recall_curve(classes_true, classes_knn)
precision_tree, recall_tree, _ = metrics.precision_recall_curve(classes_true, classes_tree)

recall_threshold = 0.7
print "\nLogreg max: ", max(precision_logreg[recall_logreg > recall_threshold])
print "SVM max: ", max(precision_svm[recall_svm > recall_threshold])
print "KNN max: ", max(precision_knn[recall_knn > recall_threshold])
print "Tree max: ", max(precision_tree[recall_tree > recall_threshold])

# Plot Precision-Recall curve
plt.clf()
plt.plot(recall_logreg, precision_logreg, label='Precision-Recall (logreg)')
plt.plot(recall_svm, precision_svm, label='Precision-Recall (svm)')
plt.plot(recall_knn, precision_knn, label='Precision-Recall (knn)')
plt.plot(recall_tree, precision_tree, label='Precision-Recall (tree)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Visualisation')
plt.legend(loc="lower left")
plt.show()



