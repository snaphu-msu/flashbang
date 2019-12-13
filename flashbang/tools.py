"""Misc. functions for general use
"""
import numpy as np


def ensure_sequence(x):
    """Ensure given object is in the form of a sequence.
    If object is scalar, return as length-1 list.

    parameters
    ----------
    x : 1D-array or scalar
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    else:
        return [x, ]


def find_nearest_idx(array, value):
    """Return idx for the array element nearest to the given value

    parameters
    ----------
    array : 1D array
        array to search
    value : float
        value to look for in array
    """
    idx = np.searchsorted(array, value)
    if np.abs(value - array[idx - 1]) < np.abs(value - array[idx]):
        return idx - 1
    else:
        return idx


def str_to_bool(string, true_options=("yes", "y", "true"),
                false_options=("no",  "n", "false")):
    """Converts string to boolean, e.g. for parsing shell input

    parameters
    ----------
    string : str or bool
        string to convert to bool (case insensitive)
    true_options : [str]
        (lowercase) strings which evaluate to True
    false_options : [str]
        (lowercase) strings which evaluate to False
    """
    if str(string).lower() in true_options:
        return True
    elif str(string).lower() in false_options:
        return False
    else:
        raise Exception(f'Undefined string for boolean conversion: {string}')
