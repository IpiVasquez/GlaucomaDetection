#!/usr/bin/env python3
from sys import stdout

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from lib.evaluator import evaluate


def grid_search(values, target, verbose=True):
    """Looks for the best param combinations for SVC."""
    stdout.write(' => Best Cs for SVM\n')

    results = pd.DataFrame(columns=['C', 'Accuracy'])
    cs = [10 ** c for c in range(-3, 2)]
    for i, c in enumerate(cs):
        if verbose:
            stdout.write(f'\r ==> SVC .... {i + 1}/{len(cs)}')

        evaluation = evaluate(SVC, values, target, params={
            'C': c,
            'kernel': 'linear'
        })
        evaluation.update({'C': c})
        results = results.append(evaluation, ignore_index=True).round(4)

    if verbose:
        stdout.write(f'\x1b[2k\r => Best SVM, {results.shape[0]} combs tested!\n')

    return results
