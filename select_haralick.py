#!/usr/bin/env python3
"""Finds the best parameters to calculate Haralick features."""
import cv2
import mahotas as mt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from lib.constants import HARALICK_NAMES
from lib import rimone, evaluator
from lib.features import extractor


def run():
    """Main handler.

    This function looks for the best distance on Haralick features on cup & 
    disc.
    """
    print(' => Reading RIMONE meta-data')
    ds = rimone.dataset()

    # Calculating for disc images
    print(' => Calculating best Haralick combination for disc images')
    results = get_best_comb(ds.discs, ds.Y, msg='disc')
    results.to_csv('results/haralick_best_comb_disc.csv', index=False)
    # Calculating for cup images
    print(' => Calculating best Haralick combination for cup images')
    results = get_best_comb(ds.cups, ds.Y, msg='cup')
    results.to_csv('results/haralick_best_comb_cup.csv', index=False)

    print(' => Done! You can check the results at `results/`')


def get_best_comb(images, target, msg=''):
    """Looks for the best distance on Haralick features."""
    results = pd.DataFrame()
    # In disc
    for dist in range(1, 4):
        print(f' ==> ({msg}) Distance {dist}/3 (all degrees + mean)', end='\r')
        haralick = extractor.get_haralick(images, dist, HARALICK_NAMES)
        values = StandardScaler().fit_transform(haralick)
        res = evaluator.evaluate(GaussianNB, values, target)
        res.update({
            'Distance': dist
        })
        results = results.append(res, ignore_index=True).round(4)
    return results


if __name__ == "__main__":
    run()
