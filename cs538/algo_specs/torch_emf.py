"""
Explicit-feedback matrix factorization with PyTorch.
"""

from scipy.stats import zipfian, loguniform, uniform, randint
from ..algorithms.torchmf import TorchMF

space = [
    # log-uniform (Zipf) distribution [5, 250]
    ('n_features', zipfian(1, 246, loc=4)),
    ('reg', loguniform(1.0e-6, 10)),
    ('lr', loguniform(1.0e-6, 1)),
    ('epochs', randint(5, 20)),
]

def default():
    return TorchMF(50)

def from_params(n_features, **kwargs):
    return TorchMF(n_features, **kwargs)
