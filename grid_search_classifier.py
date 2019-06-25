#!/usr/bin/env python3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from classifier_selectors import linear_svm, ada_boost, knn
# from lib.grid_search import get_best_params
from lib.constants import TRAIN_URI


def grid_search():
    """Main handler.

    This function looks for the best classifier & it's best combination of
    parameters.
    """

    print(' => Reading features dataset')
    df = pd.read_csv(TRAIN_URI)  # .drop('ids', axis=1)
    y = df['Diagnosis'].values
    x = df[df.columns[1:]].values
    results = ada_boost.grid_search(x, y).sort_values('Accuracy', ascending=False)
    print(results)
    results.to_csv('results/grid_ada_boost.csv')
    results = linear_svm.grid_search(x, y).sort_values('Accuracy', ascending=False)
    print(results)
    results.to_csv('results/grid_linear_svm.csv')
    results = knn.grid_search(x, y).sort_values('Accuracy', ascending=False)
    print(results)
    results.to_csv('results/grid_knn.csv', index=False)


if __name__ == "__main__":
    grid_search()
