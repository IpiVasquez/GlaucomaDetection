#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from classifier_selectors import linear_svm, ada_boost, knn
from lib.constants import TRAIN_URI
from lib.evaluator import evaluate


def grid_search():
    """Main handler.

    This function looks for the best classifier & it's best combination of
    parameters.
    """
    cols = ['Accuracy', 'BAS', 'BER', 'MCC', 'Sensibility', 'Specificity']

    all_res = pd.DataFrame(columns=(['Classifier'] + cols))
    print(' => Reading features dataset')
    df = pd.read_csv(TRAIN_URI)  # .drop('ids', axis=1)
    y = df['Diagnosis'].values
    x = df[df.columns[1:]].values

    print('\nNaive Bayes')
    results = pd.DataFrame([evaluate(GaussianNB, x, y)], columns=cols)
    print(results)
    results['Classifier'] = 'Naive Bayes'
    all_res = all_res.append(results.iloc[0], ignore_index=True)

    print('\nRandom Forest')
    results = pd.DataFrame([evaluate(RandomForestClassifier, x, y, params={'n_estimators': 50})], columns=cols)
    print(results)
    results['Classifier'] = 'Random Forest'
    all_res = all_res.append(results.iloc[0], ignore_index=True)

    print('\nAda Boost')
    results = ada_boost.grid_search(x, y).sort_values('Accuracy', ascending=False)
    print(results)
    results['Classifier'] = 'Ada Boost'
    all_res = all_res.append(results.iloc[0], ignore_index=True)

    print('\nSVC linear')
    results = linear_svm.grid_search(x, y).sort_values('Accuracy', ascending=False)
    print(results)
    results['Classifier'] = 'SVC'
    all_res = all_res.append(results.iloc[0], ignore_index=True)

    print('\nKNN')
    results = knn.grid_search(x, y).sort_values('Accuracy', ascending=False)
    print(results)
    results['Classifier'] = 'KNN'
    all_res = all_res.append(results.iloc[0], ignore_index=True)

    print('\n', all_res.sort_values('Accuracy', ascending=False))


if __name__ == "__main__":
    grid_search()
