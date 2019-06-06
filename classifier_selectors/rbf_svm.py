#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC


def get_best_params(values, target):
    """Main handler.

    This function looks for the best param combinations
    """
    print(' => Best features for SVM', end='\r')

    results = pd.DataFrame()
    cs = [10 ** c for c in range(-3, 2)]
    gammas = [10 ** g for g in range(-3, 1)]
    i = 0
    for c in cs:
        for g in gammas:
            print(f' ==> (rbf) SVC Combinations {i}/{len(cs) * len(gammas)}',
                  end='\r')
            i += 1
            scores = np.array([])
            for _ in range(5):
                perms = np.random.permutation(target.shape[0])
                y = target[perms]
                x = values[perms]
                scores = np.append(
                    scores,
                    cross_val_score(
                        SVC(kernel='rbf', C=c, gamma=g),
                        x,
                        y,
                        cv=5
                    ))
            avg_scores = scores.mean()
            results = results.append({
                'gamma': g,
                'C': c,
                'Score': avg_scores
            }, ignore_index=True).round(4)
    print(f' => (rbf) SVC Combinations {len(cs) * len(gammas)} tested')
    return results.pivot(index='C', columns='gamma', values='Score')
