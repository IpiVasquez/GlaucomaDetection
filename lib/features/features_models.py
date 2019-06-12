#!/usr/bin/env python3
"""Library for get features and models"""
import numpy as np
from lib.features import cdr
from lib import rimone
import cv2
import mahotas as mt
import pandas as pd
from lib.features import form

print(' => Reading RIMONE meta-data')
ds = rimone.dataset()

_cdr = []
_haralick_complete = []
_haralick_disc = []
_haralick_cup = []
_haralick = []
_form = []


HARALICK_NAMES = [
    'energy', 'contrast', 'correlation', 'variance', 'homogeneity', 'sum avg',
    'sum var', 'sum ent', 'entropy', 'diff var', 'diff ent', 'IC I',
    'IC II'
]
<< << << < Updated upstream
hh_disc = list(map(lambda x: 'Disc ' + x, HARALICK_NAMES))
hh_cup = list(map(lambda x: 'Cup ' + x, HARALICK_NAMES))
== == == =

HH_DISC = list(map(lambda x: 'Disc ' + x, HARALICK_NAMES))
HH_CUP = list(map(lambda x: 'Cup ' + x, HARALICK_NAMES))
HH_FULL = list(map(lambda x: 'Full ' + x, HARALICK_NAMES))

# Calculating Haralick features
# DEGREES = [0, 45, 90, 135, mean(calculated)]
# Full images => Orientation: 90 degrees, distance: 2
print(' => Calculating Haralick for the full images')
_haralick_complete = pd.DataFrame([
    mt.features.haralick(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), distance=2)[2]
    for img in ds.original_images
], columns=HH_FULL)
>>>>>> > Stashed changes

# Disc images => Orientation: 135 degrees, distance: 1
print(' => Calculating Haralick for the disc images')
_haralick_disc = pd.DataFrame([
    mt.features.haralick(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), distance=1)[3]
    for img in ds.discs
], columns=HH_DISC)

# Cup images => Orientation: 90 degrees, distance: 3
print(' => Calculating Haralick for the cup images')
_haralick_cup = pd.DataFrame([
    mt.features.haralick(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), distance=3)[2]
    for img in ds.cups
], columns=HH_CUP)

_haralick = pd.concat((_haralick_disc, _haralick_cup), axis=1)

# CDR
print(' => Calculating CDR')
_cdr = pd.DataFrame([
    cdr(disc, cup) for disc, cup in zip(ds.discs, ds.cups)
], columns=['CDR'])

# Calculating form features
cols_disc = ['Perimeter_disc', 'Area_disc', 'Compacity_disc', 'X centroid_disc', 'Y centroid_disc']
cols_cup = ['Perimeter_cup', 'Area_cup', 'Compacity_cup', 'X centroid_cup', 'Y centroid_cup']
print(' => Calculating form features')

_form_disc = []
_form_cup = []
for i in range(len(ds.disc_masks)):
    _form_disc.append(form.form_descriptors(ds.disc_masks[i]))
    _form_cup.append(form.form_descriptors(ds.cup_masks[i]))

_form_disc = pd.DataFrame(_form_disc, columns=cols_disc)
_form_cup = pd.DataFrame(_form_cup, columns=cols_cup)

_form = pd.concat((_form_disc, _form_cup), axis=1)


# DF to store features
meta = pd.DataFrame()
meta['ids'] = ds.ids
meta['Diagnosis'] = ds.Y

_models = [
    pd.concat((meta, pd.DataFrame(_cdr), pd.DataFrame(_haralick), pd.DataFrame(_form)), axis=1),
    pd.concat((meta, pd.DataFrame(_cdr)), axis=1),
    pd.concat((meta, pd.DataFrame(_haralick)), axis=1),
    pd.concat((meta, pd.DataFrame(_form)), axis=1),
    pd.concat((meta, pd.DataFrame(_cdr), pd.DataFrame(_haralick)), axis=1),
    pd.concat((meta, pd.DataFrame(_cdr), pd.DataFrame(_form)), axis=1),
    pd.concat((meta, pd.DataFrame(_haralick), pd.DataFrame(_form)), axis=1)
]


def cdr():
    return pd.concat((pd.DataFrame(_cdr), meta), axis=1)


def haralick():
    return pd.concat((pd.DataFrame(_haralick), meta), axis=1)


def haralick_disc():
    return pd.concat((pd.DataFrame(_haralick_disc), meta), axis=1)


def haralick_cup():
    return pd.concat((pd.DataFrame(_haralick_cup), meta), axis=1)


def haralick_complete():
    return pd.concat((pd.DataFrame(_haralick_complete), meta), axis=1)


def models():
    return _models
