#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lib.evaluator import evaluate


def run():
    print(' => Reading features dataset')
    df = pd.read_csv('results/processed_extracted_features.csv').drop(
        'ids', axis=1)
    y = df['Diagnosis'].values
    x = df[df.columns[1:]].values
    x = StandardScaler().fit_transform(x)
    results = pd.DataFrame(columns=['Classifier', 'Params', 'BAS', 'BER', 'MCC', 'Accuracy'])
    print(' => Testing classifiers')
    print(' ==> Naive Bayes .. ', end='\r')
    evaluation = evaluate(GaussianNB, x, y)
    evaluation.update({'Classifier': 'Naive Bayes'})
    results = results.append(evaluation, ignore_index=True).round(4)
    print(f' ==> Naive Bayes .. {evaluation["BAS"]}')
    print(' ==> Random Forest .. ', end='\r')
    evaluation = evaluate(RandomForestClassifier, x, y, params={
        'n_estimators': 50
    })
    evaluation.update({
        'Classifier': 'Random Forest',
        'Params': '# Trees: 50'
    })
    results = results.append(evaluation, ignore_index=True).round(4)
    print(f' ==> Random Forest .. {evaluation["BAS"]}')
    print(' ==> Ada Boost .. ', end='\r')
    evaluation = evaluate(AdaBoostClassifier, x, y, params={
        'learning_rate': 0.05
    })
    evaluation.update({
        'Classifier': 'Ada Boost',
        'Params': 'Learning rate: 0.5'
    })
    results = results.append(evaluation, ignore_index=True).round(4)
    print(f' ==> Ada Boost .. {evaluation["BAS"]}')
    print(' ==> SVC .. ', end='\r')
    evaluation = evaluate(SVC, x, y, params={
        'kernel': 'linear',
        'C': 0.1
    })
    evaluation.update({
        'Classifier': 'SVM',
        'Params': 'kernel: linear, C: 0.1'
    })
    results = results.append(evaluation, ignore_index=True).round(4)
    print(f' ==> SVC .. {evaluation["BAS"]}')
    print(' ==> KNN .. ', end='\r')
    evaluation = evaluate(KNeighborsClassifier, x, y, params={
        'n_neighbors': 7
    })
    evaluation.update({
        'Classifier': 'KNN',
        'Params': 'K: 7'
    })
    results = results.append(evaluation, ignore_index=True).round(4)
    print(f' ==> KNN .. {evaluation["BAS"]}')
    print(' => Done!')

    return results


if __name__ == "__main__":
    res = run()
    res.to_csv('results/test_classifiers.csv', index=False)
