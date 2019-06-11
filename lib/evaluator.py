import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneOut


def evaluate(classifier, X, Y, params={}):
    loo = LeaveOneOut()
    preds = np.array([])
    for train_idx, test_idx in loo.split(X):
        model = classifier(*params)
        model.fit(X[train_idx], Y[train_idx])
        preds = np.append(preds, model.predict(X[test_idx]))

    mcc = matthews_corrcoef(Y, preds)
    score = balanced_accuracy_score(Y, preds)

    return {
        'MCC': mcc,
        'Score': score
    }
