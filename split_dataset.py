#!/usr/bin/env python
import pandas as pd

from sklearn.model_selection import train_test_split

from lib.constants import FEATURES_URI, TRAIN_URI, TEST_URI, RANDOM_STATE


def split_dataset(file=FEATURES_URI):
    """Splits a DataSet in training & testing sets."""
    df = pd.read_csv(FEATURES_URI)
    train, test = train_test_split(df, stratify=df['Diagnosis'], test_size=0.2,
                                   random_state=RANDOM_STATE)

    return train, test


if __name__ == '__main__':
    train, test = split_dataset()
    train.to_csv(TRAIN_URI)
    test.to_csv(TEST_URI)
