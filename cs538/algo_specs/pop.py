"""
Most-Popular
"""
from scipy.stats import zipfian, loguniform, uniform, randint
from lenskit.algorithms.basic import Popular

def default():
    return Popular()