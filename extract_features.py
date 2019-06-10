#!/usr/bin/env python3
import cv2
import mahotas as mt
import pandas as pd
from lib.features import cdr
from lib import rimone
import numpy as np
from lib.features import form

RESULT_URI = 'results/extracted_features.csv'
HARALICK_NAMES = [
    'energy', 'contrast', 'correlation', 'variance', 'homogeneity', 'sum avg',
    'sum var', 'sum ent', 'entropy', 'diff var', 'diff ent', 'IC I',
    'IC II'
]


def extract_features(verbose=True):
    if verbose:
        print(' => Getting dataset')

    if verbose:
        print(' => Creating DF with information from images')

    ds = rimone.dataset()
    # DF to store features
    meta = pd.DataFrame()
    meta['ids'] = ds.ids
    meta['Diagnosis'] = ds.Y

    hh_disc = list(map(lambda x: 'Disc ' + x, HARALICK_NAMES))
    hh_cup = list(map(lambda x: 'Cup ' + x, HARALICK_NAMES))
    hh_full = list(map(lambda x: 'Full ' + x, HARALICK_NAMES))

    # Calculating features
    features = pd.DataFrame()

    # Calculating Haralick features
    # DEGREES = [0, 45, 90, 135, mean(calculated)]
    # Full images => Orientation: 90 degrees, distance: 2
    print(' => Calculating Haralick for the full images')
    haralick_features = pd.DataFrame([
        mt.features.haralick(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), distance=2)[2]
        for img in ds.original_images
    ], columns=hh_full)
    features = pd.concat((features, haralick_features), axis=1)
    # Disc images => Orientation: 135 degrees, distance: 1
    print(' => Calculating Haralick for the disc images')
    haralick_features = pd.DataFrame([
        mt.features.haralick(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), distance=1)[3]
        for img in ds.discs
    ], columns=hh_disc)
    features = pd.concat((features, haralick_features), axis=1)
    # Cup images => Orientation: 90 degrees, distance: 3
    print(' => Calculating Haralick for the cup images')
    haralick_features = pd.DataFrame([
        mt.features.haralick(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), distance=3)[2]
        for img in ds.cups
    ], columns=hh_cup)
    features = pd.concat((features, haralick_features), axis=1)
    
    
  # Joining target & ids with features
    features = pd.concat((meta, features), axis=1)

    return features


if __name__ == '__main__':
    df = extract_features()
    print(' => Saving as CSV')
    df.to_csv(RESULT_URI, index=False)
