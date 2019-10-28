# -*- coding: utf-8 -*-

__author__ = 'nyash myash'


import pandas
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

train_data = pandas.read_csv("salary-train.csv", skipinitialspace=True)
test_data = pandas.read_csv("salary-test-mini.csv", skipinitialspace=True)

# data preprocessing
train_data['FullDescription'] = train_data['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)
test_data['FullDescription'] = test_data['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)

# replace missed values to str 'nan'
train_data['LocationNormalized'].fillna('nan', inplace=True)
train_data['ContractTime'].fillna('nan', inplace=True)
test_data['LocationNormalized'].fillna('nan', inplace=True)
test_data['ContractTime'].fillna('nan', inplace=True)

# text to TF-IDF
vectorizer = TfidfVectorizer(min_df=5)
description_train_data = vectorizer.fit_transform(train_data['FullDescription'])
description_test_data = vectorizer.transform(test_data['FullDescription'])

# one-hot-coding
enc = DictVectorizer()
X_train_categ = enc.fit_transform(train_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([description_train_data, X_train_categ])
X_test = hstack([description_test_data, X_test_categ])

Y_train = train_data['SalaryNormalized']

clf = Ridge(alpha=1.0, random_state=241)
clf.fit(X_train,Y_train)

print np.round(clf.predict(X_test), 2)

#
# print common

# for i in raw_data.columns[:3]:
#     raw_data[i] = raw_data[i].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)






# enc_1 = DictVectorizer()

# loc_val = raw_data['LocationNormalized'].to_dict('records')

# X_train_categ = enc.fit_transform(raw_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
# X_test_categ = enc.transform(raw_data[['LocationNormalized', 'ContractTime']].to_dict('records'))


# print X_train_categ.shape

# x = hstack(description_data, X_train_categ)
# print x

