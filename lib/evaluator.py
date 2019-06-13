import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneOut


def evaluate(classifier, X, Y, params={}):
    loo = LeaveOneOut()
    preds = np.array([])
    for train_idx, test_idx in loo.split(X):
        model = classifier(**params)
        model.fit(X[train_idx], Y[train_idx])
        preds = np.append(preds, model.predict(X[test_idx]))

    return {
        'MCC': matthews_corrcoef(Y, preds),
        'Accuracy': (Y == preds).mean(),
        'BAS': balanced_accuracy_score(Y, preds),
        'BER': ber(Y, preds)
    }


def ber(y_true, y_predicted):
    """Calculates BER for classification."""
    b = 0
    classes = np.unique(y_true)
    # Iterate over classes
    for c in classes:
        idx = y_true == c
        c_prediction = y_predicted[idx]
        c_true = y_true[idx]
        b += (c_true != c_prediction).sum() / c_true.shape[0]
    return b / classes.shape[0]
