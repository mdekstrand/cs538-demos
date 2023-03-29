"""
Logistic matrix factorization with PyTorch (sampled).
"""

from scipy.stats import zipfian, loguniform, uniform, randint
from ..algorithms.torchmfsamp import TorchSampledMF

space = [
    # log-uniform (Zipf) distribution [5, 250]
    ('n_features', zipfian(1, 246, loc=4)),
    ('reg', loguniform(1.0e-6, 10)),
    ('lr', loguniform(1.0e-6, 1)),
    ('epochs', randint(5, 20)),
]

def default():
    return TorchSampledMF(50, loss='logistic')

def from_params(n_features, **kwargs):
    return TorchSampledMF(n_features, loss='logistic', **kwargs)
