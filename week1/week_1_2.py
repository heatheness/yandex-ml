# -*- coding: utf-8 -*-

__author__ = 'nyash myash'


import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


data = pandas.read_csv('data.csv', index_col='PassengerId')
# print data.head()
data = data[['Pclass','Fare', 'Age','Sex', 'Survived']]

data = data.dropna(subset=['Pclass','Fare', 'Age','Sex'], how='any')
# print data

data = data.replace(['male', 'female'], [1,0])

# data = data.replace('female', 0)

# data.loc[data['Sex'] == 'male', ['Sex']] = 1

# data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
# print data

# d = data.ix[data['Sex'] == 'male'] = 1
# print data


matrix_data = data.as_matrix(columns=data.columns[:])
# print matrix_data
x = matrix_data[:,0:4]
y = matrix_data[:,4]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(x, y)
print clf.feature_importances_

tree.export_graphviz(clf, out_file='tree.dot',
                     feature_names=['CabinClass', 'Fare', 'Age', 'Sex'],
                     class_names=['Dead', 'Alive'],
                     label='root',
                     rounded=True,
                     filled=True)