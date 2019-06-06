#!/usr/bin/env python3
"""Feature extraction."""
import cv2
import mahotas as mt
import pandas as pd
from lib.features import form
from lib import rimone

from sklearn.preprocessing import StandardScaler


HARALICK_NAMES = [
    'energy', 'contrast', 'correlation', 'variance', 'homogeneity', 'sum avg',
    'sum var', 'sum ent', 'entropy', 'diff var', 'diff ent', 'IC I',
    'IC II'
]
FORM_NAMES = ['perimeter', 'area', 'compacity', 'centroid x', 'centroid y']


def extract_features(verbose=True):
    """Extract all the features from the RIMONE-r3 dataset."""
    if verbose:
        print(' => Getting dataset')
    ds = rimone.dataset()
    # DF to store information of each pattern
    meta = pd.DataFrame()
    meta['ids'] = ds.ids
    meta['Diagnosis'] = ds.Y
    # Haralick names
    hh_disc = list(map(lambda x: 'Disc ' + x, HARALICK_NAMES))
    hh_cup = list(map(lambda x: 'Cup ' + x, HARALICK_NAMES))
    hh_full = list(map(lambda x: 'Full ' + x, HARALICK_NAMES))
    fh_disc = list(map(lambda x: 'Disc ' + x, FORM_NAMES))
    fh_cup = list(map(lambda x: 'Cup ' + x, FORM_NAMES))
    # Calculating features
    features = pd.DataFrame()
    # Calculating form features
    if verbose:
        print(' => Calculating form features for Disc')
    form_features = get_form(ds.disc_masks, fh_disc)
    features = pd.concat((features, form_features), axis=1, sort=True)
    if verbose:
        print(' => Calculating form features for Cup')
    form_features = get_form(ds.cup_masks, fh_cup)
    features = pd.concat((features, form_features), axis=1, sort=True)
    if verbose:
        print(' => Calculating CDR')
    features['CDR'] = features['Cup area'] / features['Disc area']
    # Calculating Haralick features
    # DEGREES = [0, 45, 90, 135, mean(calculated)]
    # Full images => Orientation: 90 degrees, distance: 2
    if verbose:
        print(' => Calculating Haralick for the full images')
    haralick_features = get_haralick(ds.original_images, 2, 2, hh_full)
    features = pd.concat((features, haralick_features), axis=1, sort=True)
    # Disc images => Orientation: 135 degrees, distance: 1
    if verbose:
        print(' => Calculating Haralick for the disc images')
    haralick_features = get_haralick(ds.discs, 1, 3, hh_disc)
    features = pd.concat((features, haralick_features), axis=1, sort=True)
    # Cup images => Orientation: 90 degrees, distance: 3
    if verbose:
        print(' => Calculating Haralick for the cup images')
    haralick_features = get_haralick(ds.discs, 3, 2, hh_cup)
    features = pd.concat((features, haralick_features), axis=1, sort=True)
    # Joining target & ids with features
    features = pd.concat((meta, features), axis=1, sort=True)

    return features


def get_haralick(imgs, distance, degree, header):
    """Gets haralick features according to the parameters received."""
    return pd.DataFrame([
        mt.features.haralick(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), distance=distance)[degree]
        for img in imgs
    ], columns=header)


def get_form(imgs, header):
    """Gets form features."""
    return pd.DataFrame([
        form.form_descriptors(img)
        for img in imgs
    ], columns=header)
