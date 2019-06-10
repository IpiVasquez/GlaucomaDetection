import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score

clf = GaussianNB() 

_score = []
_mcc = []
_ber = []
_weighted = []
for i in range(1,8):
    print(" => testing model: {}".format(i))
    data = pd.read_csv('results/models/model{}.csv'.format(i))
    data = data.values
    size = data.shape[1]
    X = data[:,3:size]
    y = data[:,2]
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
    weighted = weighted.sum()/len(weighted)
    _score.append(score)
    _mcc.append(mcc)
    _ber.append(ber)
    _weighted.append(weighted)

cols = ['accuracy', 'weighted', 'mcc', 'ber']
results = np.concatenate([_score, _weighted, _mcc, _ber]).reshape(4,7).T
results = pd.DataFrame(results, columns = cols)
print(' => Saving csv')
results.to_csv('results/models/test.csv')