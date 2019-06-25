#!/usr/bin/env python3
from sys import stdout

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from lib.evaluator import evaluate


def grid_search(values, target, verbose=True):
    """Looks for the best param combinations for KNN."""
    stdout.write(' => Best Ks for KNN\n')
    results = pd.DataFrame(columns=['K', 'Accuracy'])
    ks = [k for k in range(1, 20, 2)]
    for i, k in enumerate(ks):
        if verbose:
            stdout.write(f'\r ==> KNN .... {i + 1}/{len(ks)}')
        evaluation = evaluate(KNeighborsClassifier, values, target, params={
            'n_neighbors': k
        })
        evaluation.update({'K': k})
        results = results.append(evaluation, ignore_index=True).round(4)

    if verbose:
        stdout.write(f'\x1b[2k\r => KNN, {results.shape[0]} combs tested!\n')

    return results
