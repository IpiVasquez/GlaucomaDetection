#!/usr/bin/env python3
"""Feature extraction."""
import sys

from sklearn.preprocessing import StandardScaler

from lib.constants import FEATURES_URI
from lib.features import extractor


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
    df = extractor.extract(verbose=verbose)
    # Normalizing
    df.loc[:, df.columns[2:]] = StandardScaler().fit_transform(
        df.loc[:, df.columns[2:]])
    if not ids:
        df = df.drop(df.columns[0], axis=1)
    return df


if __name__ == '__main__':
    argn = len(sys.argv)
    if argn != 2:
        output_file = FEATURES_URI
    else:
        output_file = sys.argv[1]
    if argn > 2:
        print('''WARNING: You're sending too many params. Usage:
            
            python extract_features.py [output_file]
        ''')
    print(' => Extracting features')
    df = get_dataset(verbose=True)
    print(' => Saving extracted features')
    df.to_csv(output_file, index=False)
