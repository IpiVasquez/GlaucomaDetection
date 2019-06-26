#!/usr/bin/env python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from lib.constants import TRAIN_URI, TEST_URI


# Obtained from FFS
SELECTED_FEATURES = [
    'CDR', 'Disc variance', 'Disc sum avg', 'Cup LBP 51', 'Disc compacity',
    'Cup LBP 38', 'Cup LBP 48', 'Disc LBP 6', 'Cup LBP 93', 'Cup LBP 15',
    'Cup LBP 79', 'Cup LBP 94', 'Disc LBP 18', 'Cup LBP 69', 'Cup LBP 14',
    'Disc perimeter', 'Disc LBP 14', 'Cup LBP 34', 'Cup LBP 85', 'Cup LBP 88',
    'Cup LBP 5', 'Cup LBP 0', 'Cup compacity', 'Disc LBP 1', 'Cup LBP 107',
    'Cup LBP 55', 'Cup LBP 20', 'Cup LBP 42', 'Cup LBP 76', 'Disc LBP 12',
    'Disc LBP 0', 'Cup LBP 19', 'Cup LBP 12', 'Cup LBP 40', 'Cup LBP 43',
    'Cup LBP 74', 'Cup LBP 99', 'Disc homogeneity', 'Disc LBP 2', 'Cup LBP 78',
    'Cup LBP 70', 'Cup LBP 29', 'Cup LBP 61', 'Disc LBP 15', 'Cup LBP 1',
    'Cup LBP 17', 'Cup LBP 98', 'Cup LBP 28', 'Cup LBP 58', 'Cup LBP 87',
    'Cup LBP 46', 'Cup LBP 56', 'Cup LBP 31', 'Cup LBP 25', 'Cup LBP 9',
    'Cup LBP 13', 'Cup LBP 106', 'Cup sum avg', 'Cup LBP 2', 'Cup LBP 57',
    'Cup LBP 24', 'Cup LBP 75', 'Disc area', 'Cup LBP 77', 'Cup LBP 8',
    'Cup LBP 47', 'Cup LBP 52', 'Cup variance', 'Disc LBP 16', 'Cup LBP 59',
    'Disc centroid x', 'Cup LBP 16', 'Cup LBP 30', 'Cup LBP 92', 'Cup LBP 26',
    'Cup centroid y', 'Disc energy', 'Cup LBP 91', 'Cup LBP 65', 'Cup sum ent',
    'Cup centroid x', 'Cup LBP 83', 'Cup LBP 37', 'Disc sum ent', 'Cup LBP 32',
    'Cup perimeter', 'Cup LBP 7', 'Cup LBP 89', 'Cup LBP 45', 'Cup LBP 54',
    'Cup LBP 90', 'Cup LBP 6', 'Disc diff var', 'Cup IC I', 'Cup contrast',
    'Disc IC II', 'Disc correlation', 'Disc LBP 5', 'Cup LBP 10', 'Cup LBP 84',
    'Disc LBP 8', 'Disc LBP 19', 'Cup sum var', 'Cup LBP 73', 'Cup LBP 97',
    'Cup LBP 82', 'Cup LBP 39', 'Cup LBP 21', 'Cup LBP 72', 'Disc LBP 10',
    'Cup LBP 101', 'Cup IC II', 'Cup homogeneity', 'Cup LBP 36', 'Disc LBP 3',
    'Cup LBP 3', 'Disc sum var', 'Cup LBP 49', 'Cup LBP 86']


train = pd.read_csv(TRAIN_URI)
test = pd.read_csv(TEST_URI)

x_train = train[SELECTED_FEATURES].values
y_train = train['Diagnosis']
x_test = test[SELECTED_FEATURES].values
y_test = test['Diagnosis']


svc = SVC(C=0.1, kernel='linear')
svc.fit(x_train, y_train)
pred = svc.predict(x_test)

accuracy = y_test == pred
tp = accuracy[y_test != 0]
tn = accuracy[y_test == 0]
fp = tp.shape[0] - tp.sum()
fn = tn.shape[0] - tn.sum()
tp = tp.sum()
tn = tn.sum()


print(f'''SVM:
    Accuracy:\t\t {accuracy.mean()},
    Sensitivity:\t {accuracy[y_test != 0].sum() / y_test.sum()}
    Specificity:\t {accuracy[y_test == 0].sum() / (y_test == 0).sum()},
    BAS:\t\t {balanced_accuracy_score(y_test, pred)}
    BER:\t\t {1 - balanced_accuracy_score(y_test, pred)},
    MCC:\t\t {matthews_corrcoef(y_test, pred)}
''')

ada = AdaBoostClassifier(learning_rate=0.08)
ada.fit(x_train, y_train)
pred = ada.predict(x_test)

accuracy = y_test == pred
tp = accuracy[y_test != 0]
tn = accuracy[y_test == 0]
fp = tp.shape[0] - tp.sum()
fn = tn.shape[0] - tn.sum()
tp = tp.sum()
tn = tn.sum()


print(f'''Ada Boost:
    Accuracy:\t\t {accuracy.mean()},
    Sensitivity:\t {accuracy[y_test != 0].sum() / y_test.sum()}
    Specificity:\t {accuracy[y_test == 0].sum() / (y_test == 0).sum()},
    BAS:\t\t {balanced_accuracy_score(y_test, pred)}
    BER:\t\t {1 - balanced_accuracy_score(y_test, pred)},
    MCC:\t\t {matthews_corrcoef(y_test, pred)}
''')
