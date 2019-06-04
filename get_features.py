#!/usr/bin/env python3
import pandas as pd
import pickle
from lib.features import cdr
from lib import rimone

if __name__ == "__main__":
    print('=> Getting dataset')
    # In order to avoid re-reading all the images, a backup is created
    try:
        with open('raw_data.pkl', 'rb') as backup:
            print('=> Raw data backup found')
            data = pickle.load(backup)
    except FileNotFoundError:
        data = rimone.raw_data()
        with open('raw_data.pkl', 'wb+') as backup:
            print('=> Creating raw data backup')
            pickle.dump(data, backup)

    print('=> Creating DF')
    features = pd.DataFrame()
    features['eye_id'] = data['ids']
    features['diagnosis'] = data['Y']
    print('=> Calculating features')
    features['cdr'] = [
        cdr(imgs['disc_mask'], imgs['cup_mask']) for imgs in data['images']
    ]

    print('=> Saving as CSV')
    features.to_csv('features.csv', index=False)
