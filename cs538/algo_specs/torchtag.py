"""
Explicit-feedback matrix factorization with PyTorch.
"""

from scipy.stats import zipfian, loguniform, uniform, randint
from ..algorithms.torchtag import TorchTagMF

predicts_ratings = True

space = [
    # log-uniform (Zipf) distribution [5, 250]
    ('n_features', zipfian(1, 246, loc=4)),
    ('reg', loguniform(1.0e-6, 10)),
    ('lr', loguniform(1.0e-6, 1)),
    ('epochs', randint(5, 20)),
]

def default():
    return TorchTagMF(50)

def from_params(n_features, **kwargs):
    return TorchTagMF(n_features, **kwargs)
