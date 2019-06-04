#!/usr/bin/env python3
import mahotas as mt
import pandas as pd
import pickle
from lib.features import cdr
from lib import rimone

BACKUP_URI = 'raw_data.pkl'
RESULT_URI = 'features.csv'
HARALICK_NAMES = [
    'energy', 'contrast', 'correlation', 'variance', 'homogeneity', 'sum_avg',
    'sum_var', 'sum_ent', 'entropy', 'diff_var', 'dif_ent', 'info_corr_i',
    'info_corr_ii'
]


def extract_features(verbose=True):
    if verbose:
        print(' => Getting dataset')
    # In order to avoid re-reading all the images, a backup is created
    try:
        with open(BACKUP_URI, 'rb') as backup:
            if verbose:
                print(' => Raw data backup found')
            data = pickle.load(backup)
    except FileNotFoundError:
        data = rimone.raw_data()
        with open(BACKUP_URI, 'wb+') as backup:
            if verbose:
                print(' => Creating raw data backup')
            pickle.dump(data, backup)

    if verbose:
        print(' => Creating DF with information from images')
    meta = pd.DataFrame()
    meta['eye_id'] = data['ids']
    meta['diagnosis'] = data['Y']

    # DF to store features
    features = pd.DataFrame()
    hh_disc = list(map(lambda x: 'Disc ' + x, HARALICK_NAMES))
    hh_cup = list(map(lambda x: 'Cup ' + x, HARALICK_NAMES))
    # Calculating features
    for i, imgs in enumerate(data['images']):
        if verbose:
            print(f' => Calculating features ... {i}/{meta.shape[0]} eyes processed', end='\r')
        entry_feats = {
            'cdr': cdr(imgs['disc_mask'], imgs['cup_mask'])
        }
        entry_feats.update(dict(zip(
            HARALICK_NAMES,
            mt.features.haralick(imgs['original']).mean(axis=0)
        )))
        entry_feats.update(dict(zip(
            hh_disc,
            mt.features.haralick(imgs['original_disc']).mean(axis=0)
        )))
        entry_feats.update(dict(zip(
            hh_cup,
            mt.features.haralick(imgs['original_cup']).mean(axis=0)
        )))
        features = features.append(entry_feats, ignore_index=True)
    print(f' => Calculating features ... {meta.shape[0]} eyes processed')

    features = pd.concat((meta, features), axis=1, sort=True)
    return features


if __name__ == '__main__':
    df = extract_features()
    print(' => Saving as CSV')
    df.to_csv(RESULT_URI, index=False)
