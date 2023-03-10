"""
Run an algorithm.

Usage:
    run-algo.py [options] ALGO...

Options:
    -v, --verbose
        Use verbose logging
    --first-part N
        Start from partition N [default: 1]
"""
import sys
import logging
from pathlib import Path
import pandas as pd
from docopt import docopt

from lenskit.datasets import MovieLens
from lenskit import crossfold as xf
from lenskit.metrics.predict import rmse
from lenskit.batch import predict, recommend
from lenskit.util import clone
from lenskit.algorithms import Recommender

from algo_defs import algorithms, pred_algos

_log = logging.getLogger('run-algo')
pred_dir = Path('preds')
rec_dir = Path('recs')


def run_algo(name, train, test, i):
    algo = algorithms[name]
    # make a recommender
    algo = Recommender.adapt(algo)

    _log.info('training %s', name)
    algo.fit(train)

    if name in pred_algos:
        _log.info('generating predictions')
        preds = predict(algo, test)
        err = rmse(preds['prediction'], preds['rating'])
        _log.info('finished with RMSE %4f', err)
        preds.to_parquet(pred_dir / f'{name}-{i}-preds.parquet', index=False)

    _log.info('generating recommendations')
    recs = recommend(algo, test['user'].unique(), 20)
    recs.to_parquet(rec_dir / f'{name}-{i}-recs.parquet', index=False)


def main():
    opts = docopt(__doc__)
    # initialize logging
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(level=level)
    # turn off numba debug, it's noisy
    logging.getLogger('numba').setLevel(logging.INFO)

    # make sure we have algo dirs
    pred_dir.mkdir(exist_ok=True)
    rec_dir.mkdir(exist_ok=True)

    start = int(opts['--first-part'])

    for i in range(start, 6):
        train = pd.read_parquet(f'../data/ml-20m-split/part{i}-train.parquet')
        test = pd.read_parquet(f'../data/ml-20m-split/part{i}-test.parquet')
        for algo in opts['ALGO']:
            run_algo(algo, train, test, i)


if __name__ == '__main__':
    main()