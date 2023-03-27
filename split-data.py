"""
Split recommender system data.

Usage:
    split-data.py [options] --movielens NAME

Options:
    --movielens
        Split MovieLens rating data.
    -s SUFFIX, --split-suffix=SUFFIX
        The split suffix [default: split].
    NAME
        The name of the data set to split.
"""
import logging
from pathlib import Path
from docopt import docopt

from lenskit.datasets import MovieLens
from lenskit import crossfold as xf
import seedbank

_log = logging.getLogger('split-data')
_data_dir = Path('data')


def load_ml(name):
    ml = MovieLens(_data_dir / name)
    _log.info('reading ratings from %s', ml)
    return ml.ratings


def main(args):
    logging.basicConfig(level=logging.INFO)
    seedbank.init_file('params.yaml')

    if args['--movielens']:
        name = args['NAME']
        ratings = load_ml(name)
    else:
        raise RuntimeError('no split source specified')

    sfx = args['--split-suffix']
    split = _data_dir / f'{name}-{sfx}'
    _log.info('saving to %s', split)
    split.mkdir(exist_ok=True)

    for i, (train, test) in enumerate(xf.partition_users(ratings, 5, xf.SampleN(5))):
        i = i + 1
        _log.info('writing partition %d', i)
        train.to_parquet(split / f'part{i}-train.parquet', index=False)
        test.to_parquet(split / f'part{i}-test.parquet', index=False)


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)