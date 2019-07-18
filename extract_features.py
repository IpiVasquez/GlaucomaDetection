#!/usr/bin/env python3
"""Feature extraction."""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib import extractor
from lib.constants import FEATURES_URI, TRAIN_URI, TEST_URI, RANDOM_STATE


def get_dataset(verbose=False, ids=False):
    """Extracts & normalizes features from RIMONE-r3.

    Executes a function to extract the features from RIMONE-r3 images.

    Parameters
    __________
    verbose -- If true, prints the each step in the execution. Default false.

    Returns
    _______
    A pandas.DataFrame with the information features.
    """
    dataframe = extractor.extract_features(verbose=verbose)
    # Normalizing
    dataframe.loc[:, dataframe.columns[2:]] = StandardScaler().fit_transform(
        dataframe.loc[:, dataframe.columns[2:]])
    if not ids:
        return dataframe.drop(dataframe.columns[0], axis=1)
    return dataframe


if __name__ == '__main__':
    print(' => Extracting features')
    df = get_dataset(verbose=True)
    print(' => Saving extracted features')
    # Whole Dataset
    df.to_csv(FEATURES_URI, index=False)
    # Splitting
    train, test = train_test_split(df, stratify=df['Diagnosis'], test_size=0.2,
                                   random_state=RANDOM_STATE)
    # Train Dataset
    train.to_csv(TRAIN_URI, index=False)
    # Test Dataset
    test.to_csv(TEST_URI, index=False)
