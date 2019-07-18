#!/usr/bin/env python
"""
# Best accuracy
['CDR', 'Disc diff ent', 'Cup LBP 30', 'Cup LBP 93', 'Disc LBP 7', 'Cup LBP 79',
 'Cup LBP 94', 'Cup LBP 50', 'Disc contrast', 'Disc diff var', 'Cup LBP 99',
 'Cup LBP 17']

# Best sensibility
['CDR', 'Disc variance', 'Disc sum avg', 'Cup LBP 51', 'Disc compacity',
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

"""
from functools import reduce
from itertools import combinations
from sys import stdout

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from lib.constants import SELECTION_CRITERIA, TRAIN_URI
from lib.evaluator import evaluate

np.seterr(divide='ignore', over='ignore', under='ignore', invalid='ignore')


def FDR(df):
    """Calculates Fisher's Linear Discriminant for each column in DF."""
    grouped = df.groupby(df.columns[0])
    variances = grouped.var()  # Variances for each class
    means = grouped.mean()  # Means for each class

    def reducer(acc, pair):
        ci, cj = pair
        return acc + ((means.loc[ci] - means.loc[cj]) ** 2) / (variances.loc[ci] + variances.loc[cj])

    return reduce(reducer, combinations(df[df.columns[0]].unique(), 2), np.zeros(df.shape[1] - 1))


def ffs():
    stdout.write(' => Reading DF')
    df = pd.read_csv(TRAIN_URI)
    stdout.write('\r => Getting FDR ')
    fdr = FDR(df)
    stdout.write('\r => Initializing sets ')
    ir = {'CDR'}
    not_ir = set(df.columns[1:])
    not_ir.remove('CDR')
    Y = df['Diagnosis']
    result = pd.DataFrame(columns=['feature', 'Accuracy'])
    evaluation = evaluate(SVC, df[['CDR']].values, Y, params={
        'C': 0.1, 'kernel': 'linear'
    })
    evaluation['feature'] = 'CDR'
    result = result.append(evaluation, ignore_index=True)

    for i in range(len(not_ir)):
        features = list(ir)
        best_result = {SELECTION_CRITERIA: 0}
        for f in not_ir:
            values = df[features + [f]].values
            evaluation = evaluate(SVC, values, Y, params={
                'C': 0.1, 'kernel': 'linear'
            })
            if evaluation[SELECTION_CRITERIA] > best_result[SELECTION_CRITERIA]:
                best_result = evaluation
                best_result['feature'] = f
            elif evaluation[SELECTION_CRITERIA] == best_result[SELECTION_CRITERIA]:
                champion = best_result['feature']
                challenger = f
                if fdr[challenger] > fdr[champion]:
                    best_result = evaluation
                    best_result['feature'] = f

        result = result.append(best_result, ignore_index=True)
        not_ir.remove(best_result['feature'])
        ir.add(best_result['feature'])
        stdout.write('\r ==> %d features selected .. %0.04f sensibility' % (
            result.shape[0], best_result[SELECTION_CRITERIA]))
    stdout.write('\n')
    return result


if __name__ == "__main__":
    res = ffs()
    res.to_csv('results/ffs.csv', index=False)
