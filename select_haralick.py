#!/usr/bin/env python3
import cv2
import mahotas as mt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from lib import rimone

DEGREES = ['0', '45', '90', '135', 'mean of previous 4']


def run():
    """Main handler.

    This function looks for the best combination of haralick features using net
    search.
    """
    print(' => Reading RIMONE meta-data')
    ds = rimone.dataset()

    # Calculating for disc images
    print(' => Calculating best Haralick combination for disc images')
    results = get_best_comb([
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for img in ds.discs
    ], ds.Y, msg='disc')
    results.to_csv('results/haralick_best_comb_disc.csv')
    # Calculating for cup images
    print(' => Calculating best Haralick combination for cup images')
    results = get_best_comb([
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for img in ds.cups
    ], ds.Y, msg='cup')
    results.to_csv('results/haralick_best_comb_cup.csv')
    # Calculating for full images
    print(' => Calculating best Haralick combination for full images')
    results = get_best_comb([
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for img in ds.original_images
    ], ds.Y, msg='full')
    results.to_csv('results/haralick_best_comb_full.csv')

    print(' => Done! You can check the results at `results/`')


def get_best_comb(images, target, msg=''):
    """Net search over Haralick features.

    The net search is performed over the next params:
        degrees x distance: (0, 45, 90, 135, mean) x (1px, 2px, 3px)
    """
    results = pd.DataFrame()
    # In disc
    for dist in range(1, 4):
        print(f' ==> ({msg}) Distance {dist}/3 (all degrees + mean)', end='\r')
        haralick = np.array([
            mt.features.haralick(img, distance=dist)
            for img in images
        ])
        # Mean of 4 degrees as the 5th feature
        haralick = np.append(haralick,
                             haralick.mean(axis=1).reshape((target.shape[0], 1, 13)),
                             axis=1)
        for degree in range(5):  # 0, 45, 90, 135, means
            print(msg, end='\r')
            scores = np.array([])  # Results of cross validation
            for _ in range(20):  # 20 its on each one...
                perms = np.random.permutation(haralick.shape[0])
                x = haralick[perms, degree, :]
                y = target[perms]
                scores = np.append(
                    scores,
                    cross_val_score(
                        GaussianNB(),
                        x,
                        y,
                        cv=5
                    ))
            avg_score = scores.mean()
            results = results.append({
                'Distance': dist,
                'Degrees': DEGREES[degree],
                'Score': avg_score
            }, ignore_index=True).round(4)
    return results.pivot(index='Distance', columns='Degrees', values='Score')


if __name__ == "__main__":
    run()
