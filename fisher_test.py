"""fisher test"""

from functools import reduce
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf_names = ['Random forest', 'Adaboost', 'SVM']
clfs = [
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    SVC(kernel='linear', C=0.1)
]
# clf = GaussianNB()

data = pd.read_csv('datasets/extracted_features.csv')


def FDR(df):
    """Calculates Fisher's Linear Discriminant for each column in DF."""
    grouped = df.groupby(df.columns[0])
    variances = grouped.var()  # Variances for each class
    means = grouped.mean()  # Means for each class

    def reducer(acc, pair):
        ci, cj = pair
        return acc + ((means.loc[ci] - means.loc[cj]) ** 2) / (variances.loc[ci] + variances.loc[cj])

    return reduce(reducer, combinations(df[df.columns[0]].unique(), 2), np.zeros(df.shape[1] - 1))


target = data.values[:, 0]
print(target)
_fdr = FDR(data)
_fdr = _fdr.sort_values(ascending=False)
print(' => Saving fdr.csv')
_fdr.to_csv('results/fdr.csv')

clf_index = 0
fdr_results = []
for clf in clfs:
    print('\n{}'.format(clf_names[clf_index]))
    _score = []
    _mcc = []
    _ber = []
    _weighted = []

    ind = []
    i = 1
    for f in _fdr.index:
        ind.append(f)
        ds = data[ind]
        print(" => testing {} fdr features".format(i))
        size = data.shape[1]
        X = ds
        y = target
        y = y.astype('int')

        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        # print(y_train.sum())
        # print(y_test.sum())
        # print(y.sum())

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
        _score.append(score)
        _mcc.append(mcc)
        _ber.append(ber)
        _weighted.append(weighted)
        i += 1
    cols = ['accuracy', 'weighted', 'mcc', 'ber']
    results = np.concatenate([_score, _weighted, _mcc, _ber]).reshape(4, len(data.columns) - 1).T
    results = pd.DataFrame(results, columns=cols)
    fdr_results.append(results)
    print(' => Saving csv')
    results.to_csv('results/fdr_{}.csv'.format(clf_names[clf_index]))
    clf_index += 1

print(fdr_results)
