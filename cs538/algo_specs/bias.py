"""
User-item rating bias.
"""
from scipy.stats import zipfian, loguniform, uniform, randint
from lenskit.algorithms.basic import Bias

predicts_ratings = True

space = [
    ('uprior', loguniform(1.0e-5, 100)),
    ('iprior', loguniform(1.0e-5, 100)),
]

def default():
    return Bias()

def from_params(uprior, iprior, mrr=None):
    return Bias(damping=(uprior, iprior))
