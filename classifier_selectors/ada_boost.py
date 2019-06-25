#!/usr/bin/env python3
from sys import stdout

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

from lib.evaluator import evaluate


def grid_search(values, target, verbose=True):
    """Looks for the best param combinations for AdaBoost."""
    if verbose:
        stdout.write(' => Getting best LRates for AdaBoost\n')

    results = pd.DataFrame(columns=['Learning rate', 'Accuracy'])
    l_rates = [lr / 1000 for lr in range(80, 121, 10)]
    for i, lr in enumerate(l_rates):
        if verbose:
            stdout.write(f'\r ==> AdaBoost LRates .... {i + 1}/{len(l_rates)}')

        evaluation = evaluate(AdaBoostClassifier, values, target, params={
            'learning_rate': lr
        })
        evaluation.update({'Learning rate': lr})
        results = results.append(evaluation, ignore_index=True).round(4)

    if verbose:
        stdout.write(f'\x1b[2k\r => AdaBoost, {results.shape[0]} combs tested! \n')

    return results
