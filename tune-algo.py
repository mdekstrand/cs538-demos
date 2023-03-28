#!/usr/bin/env python3
"""
Tune hyperparameters for an algorithm.

Usage:
    tune-algo.py [options] ALGO DIR

Options:
    -v, --verbose
        Increase logging verbosity.
    -r FILE, --record=FILE
        Record individual points to FILE.
    -o FILE
        Save parameters to FILE.
    -p PART, --partition=PART
        Tune on on partition PART [default: 1].
    -n N, --num-points=N
        Test N points in hyperparameter space [default: 60].
    --rmse
        Tune on RMSE instead of MRR.
    ALGO
        The algorithm to tune.
    DIR
        The test data directory
"""

import sys
from pathlib import Path
import logging
from importlib import import_module
import json
import csv

from docopt import docopt
import pandas as pd
import numpy as np

from lenskit import batch, topn
from lenskit.metrics.predict import rmse
from lenskit.algorithms import Recommender
import seedbank

from cs538 import algo_specs

_log = logging.getLogger('tune-algo')
metric = 'MRR'


def sample(space, state):
    "Sample a single point from a search space."
    return {
        name: dist.rvs(random_state=state)
        for (name, dist) in space
    }


def evaluate(point):
    "Evaluate the algorithm with a set of parameters."
    algo = algo_mod.from_params(**point)
    _log.info('evaluating %s', algo)

    if metric == 'RMSE':
        algo.fit(train_data)
        preds = batch.predict(algo, test_data)
        errs = preds['prediction'] - preds['rating']
        # assume missing values are completely off (5 star difference)
        errs = errs.fillna(5)
        return np.mean(np.square(errs))
    else:
        algo = Recommender.adapt(algo)
        
        recs = batch.recommend(algo, test_users, 5000)
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.recip_rank, k=5000)
        scores = rla.compute(recs, test_data, include_missing=True)
        mrr = scores['recip_rank'].fillna(0).mean()
        return mrr

def main(args):
    global algo_mod, train_data, test_data, test_users, metric
    level = logging.DEBUG if args['--verbose'] else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    logging.getLogger('numba').setLevel(logging.INFO)

    seedbank.init_file('params.yaml')

    algo_name = args['ALGO']
    _log.info('loading algorithm %s', algo_name)
    algo_mod = algo_specs.algorithms[algo_name]

    data = Path(args['DIR'])
    _log.info('loading data from %s', data)
    part = int(args['--partition'])
    train_data = pd.read_parquet(data / f'part{part}-train.parquet')
    test_data = pd.read_parquet(data / f'part{part}-test.parquet')
    test_users = test_data['user'].unique()

    state = seedbank.numpy_random_state()

    points = []
    record_fn = args['--record']
    if record_fn:
        rcols = [name for (name, _dist) in algo_mod.space]
        rcols.append('mrr')
        recfile = open(record_fn, 'w')
        record = csv.DictWriter(recfile, rcols)
        record.writeheader()
    else:
        record = None

    if args['--rmse']:
        _log.info('scoring predictions on RMSE')
        metric = 'RMSE'

    npts = int(args['--num-points'])
    _log.info('evaluating at %d points', npts)
    for i in range(npts):
        point = sample(algo_mod.space, state)
        _log.info('iter %d: %s', i + 1, point)
        value = evaluate(point)
        _log.info('iter %d: %s=%0.4f', i + 1, metric, value)
        point[metric] = value
        points.append(point)
        if record:
            record.writerow(point)
            recfile.flush()

    points = sorted(points, key=lambda p: p[metric], reverse=(metric != 'RMSE'))
    best_point = points[0]
    _log.info('finished in with %s %.3f', metric, best_point[metric])
    for p, v in best_point.items():
        _log.info('best %s: %s', p, v)

    if record:
        recfile.close()

    fn = args.get('-o', None)
    if fn:
        _log.info('saving params to %s', fn)
        Path(fn).write_text(json.dumps(best_point))


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
