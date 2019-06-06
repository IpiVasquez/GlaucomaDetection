#!/usr/bin/env python3
"""Feature extraction."""
from sklearn.preprocessing import StandardScaler

from lib.features import extractor

RAW_RESULT_URI = 'results/raw_extracted_features.csv'
PROCESSED_RESULT_URI = 'results/processed_extracted_features.csv'


if __name__ == '__main__':
    df = extractor.extract_features(verbose=True)
    print(' => Saving raw features in CSV')
    df.to_csv(RAW_RESULT_URI, index=False)
    df.loc[:, df.columns[2:]] = StandardScaler().fit_transform(
        df.loc[:, df.columns[2:]])
    df.to_csv(PROCESSED_RESULT_URI, index=False)
