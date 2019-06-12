#!/usr/bin/env python3
"""Feature extraction."""
import cv2
import mahotas as mt
import pandas as pd

from lib.constants import HH_DISC, HH_CUP, FH_DISC, FH_CUP
from lib.features import form
from lib import rimone


def extract_features(verbose=True):
    """Extract all the features from the RIMONE-r3 dataset."""
    if verbose:
        print(' => Getting dataset')
    ds = rimone.dataset()
    # DF to store information of each pattern
    meta = pd.DataFrame()
    meta['ids'] = ds.ids
    meta['Diagnosis'] = ds.Y
    # Calculating features
    features = pd.DataFrame()
    # Calculating form features
    if verbose:
        print(' => Calculating form features for Disc')
    form_features = get_form(ds.disc_masks, FH_DISC)
    features = pd.concat((features, form_features), axis=1, sort=True)
    if verbose:
        print(' => Calculating form features for Cup')
    form_features = get_form(ds.cup_masks, FH_CUP)
    features = pd.concat((features, form_features), axis=1, sort=True)
    if verbose:
        print(' => Calculating CDR')
    features['CDR'] = features['Cup area'] / features['Disc area']
    # Calculating Haralick features
    # Disc images => Orientation: 135 degrees, distance: 1
    if verbose:
        print(' => Calculating Haralick for the disc images')
    haralick_features = get_haralick(ds.discs, 1, HH_DISC)
    features = pd.concat((features, haralick_features), axis=1, sort=True)
    # Cup images => Orientation: 90 degrees, distance: 3
    if verbose:
        print(' => Calculating Haralick for the cup images')
    haralick_features = get_haralick(ds.discs, 3, HH_CUP)
    features = pd.concat((features, haralick_features), axis=1, sort=True)
    # Joining target & ids with features
    features = pd.concat((meta, features), axis=1, sort=True)

    return features


def get_haralick(imgs, distance, header):
    """Gets haralick features according to the parameters received."""
    return pd.DataFrame([
        mt.features.haralick(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            distance=distance, ignore_zeros=True, return_mean=True
        )
        for img in imgs
    ], columns=header)


def get_form(imgs, header):
    """Gets form features."""
    return pd.DataFrame([
        form.form_descriptors(img)
        for img in imgs
    ], columns=header)
