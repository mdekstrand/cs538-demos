"""
Lift.
"""
from scipy.stats import zipfian, loguniform, uniform, randint
from ..algorithms.lift import Lift

def default():
    return Lift()