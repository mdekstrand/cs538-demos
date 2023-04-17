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
    --points-only
        Print the points that would be used, without testing.
    ALGO
        The algorithm to tune.
    DIR
        The test data directory
"""

import sys
from pathlib import Path
import logging
from dataclasses import dataclass
import json
import csv

from docopt import docopt
import pandas as pd
import numpy as np

from lenskit import batch, topn
from lenskit.algorithms import Recommender
from lenskit.util.parallel import is_mp_worker
from lenskit.util import Stopwatch
import seedbank

from cs538 import algo_specs

_log = logging.getLogger('tune-algo')


if is_mp_worker():
    # disable torch threading in worker proceses
    import torch
    torch.set_num_threads(1)


@dataclass
class TuneContext:
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    test_users: np.ndarray
    metric: str = 'MRR'


def sample(space, state):
    "Sample a single point from a search space."
    return {
        name: dist.rvs(random_state=state)
        for (name, dist) in space
    }


def evaluate(ctx: TuneContext, point):
    "Evaluate an algorithm."
    algo = algo_mod.from_params(**point)
    _log.info('evaluating %s', algo)

    if ctx.metric == 'RMSE':
        ttime = Stopwatch()
        algo.fit(ctx.train_data)
        ttime.stop()
        _log.info('trained model in %s', ttime)

        rtime = Stopwatch()
        preds = batch.predict(algo, ctx.test_data)
        rtime.stop()

        errs = preds['prediction'] - preds['rating']
        # assume missing values are completely off (5 star difference)
        errs = errs.fillna(5)
        val = np.mean(np.square(errs))
    else:
        algo = Recommender.adapt(algo)
        ttime = Stopwatch()
        algo.fit(ctx.train_data)
        ttime.stop()
        _log.info('trained model in %s', ttime)

        rtime = Stopwatch()
        recs = batch.recommend(algo, ctx.test_users, 5000)
        rtime.stop()
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.recip_rank, k=5000)
        scores = rla.compute(recs, ctx.test_data, include_missing=True)
        val = scores['recip_rank'].fillna(0).mean()
    
    return {
        ctx.metric: val,
        'TrainTime': ttime.elapsed(),
        'RunTime': rtime.elapsed(),
    }


def main(args):
    global algo_mod
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

    ctx = TuneContext(train_data, test_data, test_users)

    state = seedbank.numpy_random_state()

    if args['--rmse']:
        _log.info('scoring predictions on RMSE')
        ctx.metric = 'RMSE'

    record_fn = args['--record']
    if record_fn:
        rcols = [name for (name, _dist) in algo_mod.space]
        rcols += [ctx.metric, 'TrainTime', 'RunTime']
        recfile = open(record_fn, 'w')
        record = csv.DictWriter(recfile, rcols)
        record.writeheader()
    else:
        record = None

    npts = int(args['--num-points'])
    _log.info('evaluating at %d points', npts)
    for i in range(npts):
        point = sample(algo_mod.space, state)
        _log.info('point %d: %s', i, point)
        if args['--points-only']:
            continue

        res = evaluate(ctx, point)
        _log.info('%s: %s=%.4f', point, ctx.metric, res[ctx.metric])
        point.update(res)
        if record:
            record.writerow(point)
            recfile.flush()

    points = sorted(points, key=lambda p: p[ctx.metric], reverse=(ctx.metric != 'RMSE'))
    best_point = points[0]
    _log.info('finished in with %s %.3f', ctx.metric, best_point[ctx.metric])
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
