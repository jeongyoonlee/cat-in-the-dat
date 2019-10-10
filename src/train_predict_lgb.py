#!/usr/bin/env python

import argparse
import lightgbm as lgb
import logging
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time

from kaggler.metrics import auc
from kaggler.data_io import load_data

from const import N_FOLD, SEED, N_JOB

np.random.seed(SEED)


def train_predict(train_file, test_file, feature_map_file, predict_valid_file, predict_test_file,
                  feature_importance_file, n_est=100, n_leaf=200, lrate=.1, n_min=8, subcol=.3, subrow=.8,
                  subrow_freq=100, n_stop=100, retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.info('{}'.format(model_name))
    logging.info(('n_est={}, n_leaf={}, lrate={}, '
                  'n_min={}, subcol={}, subrow={},'
                  'subrow_freq={}, n_stop={}').format(n_est, n_leaf, lrate, n_min,
                                                      subcol, subrow, subrow_freq, n_stop))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    params = {'boosting_type': 'gbdt',
              'objective': 'binary',
              'num_leaves': n_leaf,
              'learning_rate': lrate,
              'feature_fraction': subcol,
              'bagging_fraction': subrow,
              'bagging_freq': subrow_freq,
              'min_data_in_leaf': n_min,
              'feature_fraction_seed': SEED,
              'bagging_seed': SEED,
              'data_random_seed': SEED,
              'metric': 'auc',
              'verbose': 0,
              'num_threads': N_JOB}

    logging.info('Loading CV Ids')
    cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    p_val = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    feature_name, feature_ext = os.path.splitext(train_file)
    feature_name = os.path.splitext(feature_name)[0]

    for i, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
        logging.info('Training model #{}'.format(i))
        cv_train_file = '{}.trn{}{}'.format(feature_name, i, feature_ext)
        cv_test_file = '{}.tst{}{}'.format(feature_name, i, feature_ext)

        if os.path.exists(cv_train_file):
            is_cv_feature = True
            X_cv, _ = load_data(cv_train_file)
            X_tst_cv, _ = load_data(cv_test_file)

            lgb_trn = lgb.Dataset(np.hstack((X[i_trn], X_cv[i_trn])), y[i_trn])
            lgb_val = lgb.Dataset(np.hstack((X[i_val], X_cv[i_val])), y[i_val])
        else:
            is_cv_feature = False
            lgb_trn = lgb.Dataset(X[i_trn], y[i_trn])
            lgb_val = lgb.Dataset(X[i_val], y[i_val])

        if i == 1:
            logging.info('Training with early stopping')
            clf = lgb.train(params,
                            lgb_trn,
                            num_boost_round=n_est,
                            early_stopping_rounds=n_stop,
                            valid_sets=lgb_val,
                            verbose_eval=100)

            n_best = clf.best_iteration
            logging.info('best iteration={}'.format(n_best))

            df = pd.read_csv(feature_map_file, sep='\t', names=['id', 'name', 'type'])
            df['gain'] = clf.feature_importance(importance_type='gain', iteration=n_best)
            df.loc[:, 'gain'] = df.gain / df.gain.sum()
            df.sort_values('gain', ascending=False, inplace=True)
            df.to_csv(feature_importance_file, index=False)
            logging.info('feature importance is saved in {}'.format(feature_importance_file))
        else:
            clf = lgb.train(params,
                            lgb_trn,
                            num_boost_round=n_best,
                            valid_sets=lgb_val,
                            verbose_eval=100)

        if is_cv_feature:
            p_val[i_val] = clf.predict(np.hstack((X[i_val], X_cv[i_val])))
        else:
            p_val[i_val] = clf.predict(X[i_val])

        logging.info('CV #{}: {:.6f}'.format(i, auc(y[i_val], p_val[i_val])))

        if not retrain:
            if is_cv_feature:
                p_tst += clf.predict(np.hstack((X_tst, X_tst_cv))) / N_FOLD
            else:
                p_tst += clf.predict(X_tst) / N_FOLD

    logging.info('CV: {:.6f}'.format(auc(y, p_val)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        lgb_trn = lgb.Dataset(X, y)
        clf = lgb.train(params,
                        lgb_trn,
                        num_boost_round=n_best,
                        verbose_eval=100)

        p_tst = clf.predict(X_tst)

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--feature-importance-file', required=True, dest='feature_importance_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--n-leaf', type=int, dest='n_leaf')
    parser.add_argument('--lrate', type=float)
    parser.add_argument('--subcol', type=float, default=1)
    parser.add_argument('--subrow', type=float, default=.5)
    parser.add_argument('--subrow-freq', type=int, default=100, dest='subrow_freq')
    parser.add_argument('--n-min', type=int, default=1, dest='n_min')
    parser.add_argument('--early-stop', type=int, dest='n_stop')
    parser.add_argument('--retrain', default=False, action='store_true')
    parser.add_argument('--log-file', required=True, dest='log_file')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename=args.log_file,
                        datefmt='%Y-%m-%d %H:%M:%S')

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  feature_map_file=args.feature_map_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  feature_importance_file=args.feature_importance_file,
                  n_est=args.n_est,
                  n_leaf=args.n_leaf,
                  lrate=args.lrate,
                  n_min=args.n_min,
                  subcol=args.subcol,
                  subrow=args.subrow,
                  subrow_freq=args.subrow_freq,
                  n_stop=args.n_stop,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) / 60))
