#!/usr/bin/env python

import argparse
import logging
import numpy as np
import os
import pandas as pd
import time
import warnings
from kaggler.preprocessing import LabelEncoder
from kaggler.data_io import save_data

from const import TARGET_COL, ID_COL


warnings.filterwarnings("ignore")


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file)
    tst = pd.read_csv(test_file)

    y = trn[TARGET_COL]
    n_trn = trn.shape[0]

    features = [x for x in trn.columns if x not in [ID_COL, TARGET_COL]]

    df = pd.concat([trn.drop([TARGET_COL, ID_COL], axis=1), tst.drop(ID_COL, axis=1)], axis=0)

    logging.info('label encoding')
    lbe = LabelEncoder(min_obs=50)
    df[features] = lbe.fit_transform(df[features])

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(features):
            f.write('{}\t{}\tint\n'.format(i, col))

    logging.info('saving features')
    save_data(df.values[:n_trn], y.values, train_feature_file)
    save_data(df.values[n_trn:], None, test_feature_file)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.train_feature_file,
                     args.test_feature_file,
                     args.feature_map_file)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))
