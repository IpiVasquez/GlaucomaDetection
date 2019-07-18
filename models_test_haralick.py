import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

clf = GaussianNB()
i = 3

print(" => testing model: {}".format(i))
data = pd.read_csv('results/models/model{}.csv'.format(i))
data = data.values
size = data.shape[1]
X = data[:, 3:size]
print(X.shape)
y = data[:, 2]
y = y.astype('int')
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
mcc = matthews_corrcoef(y_test, y_pred)
ber = 1 - balanced_accuracy_score(y_test, y_pred)
weighted = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring='f1_weighted'
)
weighted = weighted.sum() / len(weighted)

cols = ['accuracy', 'weighted', 'mcc', 'ber']
results = np.array([1 - ber, weighted, mcc, ber])

print(results)

X = data[:, 3:16]
print(X.shape)
y = data[:, 2]
y = y.astype('int')
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
mcc = matthews_corrcoef(y_test, y_pred)
ber = 1 - balanced_accuracy_score(y_test, y_pred)
weighted = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring='f1_weighted'
)
weighted = weighted.sum() / len(weighted)

cols = ['accuracy', 'weighted', 'mcc', 'ber']
results = np.array([1 - ber, weighted, mcc, ber])

print(results)

X = data[:, 16:29]
print(X.shape)
y = data[:, 2]
y = y.astype('int')
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
mcc = matthews_corrcoef(y_test, y_pred)
ber = 1 - balanced_accuracy_score(y_test, y_pred)
weighted = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring='f1_weighted'
)
weighted = weighted.sum() / len(weighted)

cols = ['accuracy', 'weighted', 'mcc', 'ber']
results = np.array([1 - ber, weighted, mcc, ber])

print(results)

X = data[:, 29:42]
print(X.shape)
y = data[:, 2]
y = y.astype('int')
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
mcc = matthews_corrcoef(y_test, y_pred)
ber = 1 - balanced_accuracy_score(y_test, y_pred)
weighted = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring='f1_weighted'
)
weighted = weighted.sum() / len(weighted)

cols = ['accuracy', 'weighted', 'mcc', 'ber']
results = np.array([1 - ber, weighted, mcc, ber])

print(results)
