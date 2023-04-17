#!/usr/bin/env python3
"""
Run an algorithm.

Usage:
    run-algo.py [options] ALGO

Options:
    -d DIR, --data=DIR
        Use data in DIR [default: data/ml-20m-split].
    --tags DIR
        Use tag data from DIR in addition to ratings.
    -p PFX, --prefix=PREFIX
        Prefix output dirs with PFX.
    -j N, --procs N
        Use N processes for prediction / recommendation.
    -v, --verbose
        Use verbose logging.
    --first-part N
        Start from partition N [default: 1].
    --params FILE
        Load parameters from FILE.
"""
import json
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
from lenskit.util.parallel import is_mp_worker

from cs538.algo_specs import algorithms

_log = logging.getLogger('run-algo')
pred_base = Path('preds')
rec_base = Path('recs')

if is_mp_worker():
    # disable torch threading in worker proceses
    import torch
    torch.set_num_threads(1)


def run_algo(name, algo, want_preds, pfx, train, test, i, extra):
    # make a recommender
    algo = Recommender.adapt(algo)

    _log.info('training %s', name)
    algo.fit(train, **extra)

    dname = f'{pfx}-{name}' if pfx else name

    if want_preds:
        _log.info('generating predictions')
        preds = predict(algo, test, n_jobs=n_jobs)
        err = rmse(preds['prediction'], preds['rating'])
        _log.info('finished with RMSE %4f', err)
        pred_dir = pred_base / dname
        pred_dir.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(pred_dir / f'part{i}-preds.parquet', index=False)

    _log.info('generating recommendations')
    recs = recommend(algo, test['user'].unique(), 20, n_jobs=n_jobs)
    rec_dir = rec_base / dname
    rec_dir.mkdir(parents=True, exist_ok=True)
    recs.to_parquet(rec_dir / f'part{i}-recs.parquet', index=False)


def main():
    global n_jobs
    opts = docopt(__doc__)
    # initialize logging
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(level=level)
    # turn off numba debug, it's noisy
    logging.getLogger('numba').setLevel(logging.INFO)

    data = opts['--data']
    start = int(opts['--first-part'])
    n_jobs = opts['--procs']
    if n_jobs:
        n_jobs = int(n_jobs)

    algo_name = opts['ALGO']
    _log.info('using algorithm %s', algo_name)
    algo_mod = algorithms[algo_name]
    pfn = opts.get('--params', None)
    if pfn:
        _log.info('using parameters from %s')
        params = json.loads(Path(pfn).read_text())
        algo = algo_mod.from_params(**params)
    else:
        algo = algo_mod.default()
    want_preds = getattr(algo_mod, 'predicts_ratings', False)

    extra = {}
    tag_src = opts.get('--tags', None)
    if tag_src:
        _log.info('loading tags from %s', tag_src)
        ml = MovieLens(tag_src)
        extra['tags'] = ml.tags

    for i in range(start, 6):
        train = pd.read_parquet(f'{data}/part{i}-train.parquet')
        test = pd.read_parquet(f'{data}/part{i}-test.parquet')
        run_algo(algo_name, algo, want_preds, opts['--prefix'], train, test, i, extra)


if __name__ == '__main__':
    main()
