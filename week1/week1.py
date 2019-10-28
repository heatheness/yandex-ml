# -*- coding: utf-8 -*-

__author__ = 'nyash myash'

import pandas


data = pandas.read_csv('data.csv', index_col='PassengerId')

# print data.head()

print data['Sex'].value_counts()
print data['Survived'].value_counts()

print data['Survived'].sum()/float(data['Survived'].count()) * 100

print data['Pclass'].value_counts().sum()


print float(data.loc[data['Pclass'] == 1]['Pclass'].count()) / data['Pclass'].value_counts().sum() * 100
print data.loc[data['Pclass'] == 1]['Pclass'].value_counts()/ data['Pclass'].value_counts().sum() * 100

print data['Age'].describe()

print data['SibSp'].corr(data['Parch'], method='pearson')

names = data.loc[data['Sex'] == 'female']['Name']
first_names = []

for item in names:
    if '(' in item:
            start = item.index('(') + 1
            end = item.index(')')
            first_names.append(item[start:end].split()[0])
    else:
         first_names.append(item.split(',')[1].split()[1])
    # else:
    #      print item


print first_names

n = pandas.Series(first_names)
print n.value_counts()
    # if "Miss." in item:
    #     print item.split(',')[1].split()[1]
