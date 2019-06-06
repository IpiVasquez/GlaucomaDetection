#!/usr/bin/env python3
import pandas as pd
from classifier_selectors import rbf_svm, linear_svm


def run():
    """Main handler.

    This function looks for the best classifier & it's best parameter
    combination.
    """
    print(' => Reading features dataset')
    df = pd.read_csv('results/extracted_features.csv').drop('ids', axis=1)
    y = df['Diagnosis'].values
    x = df[df.columns[1:]].values
    results = rbf_svm.get_best_params(x, y)
    results.to_csv('results/grid_rbf_svm.csv')
    results = linear_svm.get_best_params(x, y)
    results.to_csv('results/grid_linear_svm.csv')


if __name__ == "__main__":
    run()
