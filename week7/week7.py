# -*- coding: utf-8 -*-

"""
Predict Dota 2 match results based on first 5 mins macth data.
Using 2 approaches with various data improvements to compare.
Task description - in final-statement.html.
"""

__author__ = 'nyash myash'


import datetime
import numpy as np
import pandas
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale


# GRADIENT BOOSTING


def bag_of_words(hero_ids, features):
    """one hot encoding"""

    x_pick = np.zeros((features.shape[0], max(hero_ids)))

    for i, match_id in enumerate(features.index):
        for p in xrange(5):
            x_pick[i, features.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            x_pick[i, features.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    return x_pick


def predict(clf, x, y):
    """Scoring function for cross-validation"""

    proba = clf.predict_proba(x)[:, 1]
    return roc_auc_score(y, proba)

def log_reg(x, y, kf):

    c_grid = np.linspace(0.001, 1.0, num=10)
    best_score = 0
    best_c = 0
    corresponding_duration = 0

    for c in c_grid:
        start_time = datetime.datetime.now()
        clf = LogisticRegression(penalty='l2', C=c)
        cvs = cross_val_score(clf, x, y, scoring=predict, cv=kf)
        score = np.round(np.mean(cvs), 5)
        duration = (datetime.datetime.now() - start_time).total_seconds()
        if score > best_score:
            best_score = score
            best_c = np.round(c,2)
            corresponding_duration = duration

    return [best_score, best_c, corresponding_duration]

features = pandas.read_csv('features.csv', index_col='match_id')

# Get match results and create results vector
y = features['radiant_win'].values

# Drop columns connected with match results
res_data = [u'duration', u'radiant_win', u'tower_status_radiant', u'tower_status_dire', u'barracks_status_radiant',
            u'barracks_status_dire']
features.drop(res_data, axis=1, inplace=True)

features.drop('start_time', axis=1, inplace=True) # dropping to avoid warnings when using logistic regression

# Get names of columns with NaN values
f = features.count()
na_features = f[f != features.shape[0]].index.values
print "Features names with NaN\n"
for item in na_features:
    print item

# Fill in NaN
features.fillna(value=0, inplace=True)

# Create matrix of features
x = features.values

print "\n\nGRADIENT BOOSTING\n\n"
print "Building gradient boosting classifier\n"

# Build classifier and make cross-validation with AUC-ROC score
kf = KFold(len(x), n_folds=5, shuffle=True)  # Cross-validation iterator
n_trees = [10, 20, 30]
for i in n_trees:
    start_time = datetime.datetime.now()
    clf = GradientBoostingClassifier(n_estimators=i)
    cvs = cross_val_score(clf, x, y, scoring=predict, cv=kf)
    score = np.round(np.mean(cvs), 4)
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print "\nNumber of Trees: {0} \nScore: {1} \nTime to build classifier: {2:.2f} sec".format(i, score, duration)

"""
Подход 1: градиентный бустинг "в лоб"

1. Какие признаки имеют пропуски среди своих значений?
Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?

Ответ:

Признаки, имеющие пропуски:
- first_blood_time
- first_blood_team
- first_blood_player1
- first_blood_player2
- radiant_bottle_time
- radiant_courier_time
- radiant_flying_courier_time
- radiant_first_ward_time
- dire_bottle_time
- dire_courier_time
- dire_flying_courier_time
- dire_first_ward_time

Что могут означать пропуски:
first_blood_time - за первые 5 мин игры событие не произошло
radiant_bottle_time - за время игры ни одна команда не приобрела предмет "bottle"

2. Как называется столбец, содержащий целевую переменную?

Ответ: radiant_win

3. Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Какое качество при этом получилось?

Ответ:

Время кросс-валидации: 269 секунд
Качество: 0.69


4. Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге?
Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?

Ответ:

Использовать больше 30 деревьев смысла нет: качество по сравннию с 10 деревьями выросло незначительно (0.67 для 10
деревьев и 0.69 для 30), а время обучения возрасло более чем в 3 раза (88 сек для 10 деревьев и 269 сек для 30)

Для увеличения скорости при увеличении количества деревьев можно:
- использовать понижение размерности пространста признаков
- производить обучение только на части объектов
- уменьшить глубину деревьев до 2 (по умолчанию глубина - 3)
- использовать более производительную машину)

"""


# LOGISTIC REGRESSION

# 1. Building classifier w/ dropping castegories info

print "\n\nLOGISTIC REGRESSION\n\n"
print "Building classifier w/ dropping categories info\n"

x_1_scaled = scale(x)
best_score, best_c, corresponding_duration = log_reg(x_1_scaled, y, kf)
print "Best C: {}, \nbest score {}, \nbest time {}\n".format(best_c, best_score, corresponding_duration)

# 2. Building classifier with dropping categories info

heroes = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
                  'd4_hero', 'd5_hero']

categ_features =['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
                  'd4_hero', 'd5_hero', 'lobby_type']

print "\nBuilding classifier with dropping categories info\n"
dropped_categ_features = features.drop(categ_features, axis=1)
x_2 = dropped_categ_features.values
x_2_scaled = scale(x_2)
best_score, best_c, corresponding_duration = log_reg(x_2_scaled, y, kf)
print "Best C: {}, \nbest score {}, \nbest time {}\n".format(best_c, best_score, corresponding_duration)


# 3. Building classifier with  with one hot encoding


# Getting heroes id

hero_ids = set()
for hero in heroes:
    hero_ids.update(features[hero].unique())

n = len(hero_ids)
hero_ids = list(hero_ids)

x_pick = bag_of_words(hero_ids, features)
x_3_scaled = np.hstack((x_2_scaled, x_pick))


print "\nBuilding classifier with one hot encoding \n"
best_score, best_c, corresponding_duration = log_reg(x_3_scaled, y, kf)
print "Best C: {}, \nbest score {}, \nbest time {}\n".format(best_c, best_score, corresponding_duration)


# Running best classifier on test data
# Logistic regression with one hot encoding of words showed best AUC-ROC Score

print "\n\nRunning best classifier on test data\n\n"
print"Logistic regression with one hot encoding  on test data\n"

# Fit classifier

final_clf = LogisticRegression(penalty='l2', C=0.11)
final_clf.fit(x_3_scaled, y)

# Preprocess test data

features_test = pandas.read_csv('features_test.csv', index_col='match_id')
features_test.drop('start_time', axis=1, inplace=True)
features_test.fillna(value=0, inplace=True)
dropped_categ_features_test = features_test.drop(categ_features, axis=1)
x_test_values = dropped_categ_features_test.values
x_test_scaled = scale(x_test_values)
x_pick_test = bag_of_words(hero_ids, features_test)
x_test = np.hstack((x_test_scaled, x_pick_test))


# Get predictions
res = final_clf.predict_proba(x_test)

print "Predict min", np.amin(res)
print "Predict max", np.amax(res)

with open('predictions_proba.csv', 'w') as f:
    f.write('match_id,radiant_win\n')
    for i, match_id in enumerate(features_test.index):
        f.write('{},{}\n'.format(match_id,res[i][0]))


"""
Подход 2: логистическая регрессия

1. Какое качество получилось у логистической регрессии над всеми исходными признаками?
Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу?
Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

Ответ:

Качество - 0.7163.
Выше градиентного бустинга на 0.035. Разницу можно объяснить масштабированием признаков
Значительно быстрее.


2.Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)?
Чем вы можете объяснить это изменение?

Ответ:

Не влияет. Пока категориальные признаки не преобразованы к мешку слов, они, очевидно, не участвуют в построении
классификатора.

Новое качество - 0.7163


3. Сколько различных идентификаторов героев существует в данной игре?

Ответ:

108 идентификаторов. Значение максимального - 112. По каким-то причинам некоторые значения пропущены.


4. Какое получилось качество при добавлении "мешка слов" по героям?
Улучшилось ли оно по сравнению с предыдущим вариантом? Чем вы можете это объяснить?

Ответ:

Новое качество - 0.7516
Качество выросло на 0.0353
Теперь категориальные признаки участвовали в построении классификатора


5. Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?

Ответ:

Min -  0.00337739165456
Max -  0.996622608345

 """

