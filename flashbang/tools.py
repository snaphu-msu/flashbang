import numpy as np

"""
Misc. functions for general use
"""


def ensure_sequence(x):
    """Ensure given object is in the form of a sequence

    If object is scalar, return as length-1 list
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    else:
        return [x, ]


def find_nearest_idx(array, value):
    """Return idx for the array element nearest to the given value
    """
    idx = np.searchsorted(array, value)
    if np.abs(value - array[idx - 1]) < np.abs(value - array[idx]):
        return idx - 1
    else:
        return idx
