"""
Explicit-feedback matrix factorization with PyTorch.
"""

from scipy.stats import zipfian, loguniform, uniform, randint
from lenskit.algorithms.als import BiasedMF

space = [
    # log-uniform (Zipf) distribution [5, 250]
    ('n_features', zipfian(1, 246, loc=4)),
    ('uprior', loguniform(1.0e-6, 10)),
    ('iprior', loguniform(1.0e-6, 10)),
    ('ureg', loguniform(1.0e-6, 10)),
    ('ireg', loguniform(1.0e-6, 10)),
    ('epochs', randint(5, 40)),
]

def default():
    return BiasedMF(50)

def from_params(n_features, uprior, iprior, ureg, ireg, epochs, **kwargs):
    return BiasedMF(n_features, iterations=epochs, reg=(ureg, ireg), damping=(uprior, iprior))
