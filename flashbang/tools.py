"""Misc. functions for general use
"""
import numpy as np


def ensure_sequence(x, n=1):
    """Ensure given object is in the form of a sequence.
    If object is scalar, return as length-n list.

    parameters
    ----------
    x : array or scalar
    n : length of list to return
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    else:
        return n * [x]


def find_nearest_idx(array, value):
    """Return idx for the array element nearest to the given value

    Note: array assumed to be monotonically increasing (not enforced),
          will use the first element that exceeds the given value

    parameters
    ----------
    array : array
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
    """Convert string to boolean, e.g. for parsing shell input

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


def get_missing_elements(elements, array):
    """Return the elements which are missing from an array

    Returns: 1D array

    parameters
    ----------
    elements : array
        elements to look for
    array : array
        array to look in for elements
    """
    elements = np.array(elements)
    is_in = np.isin(elements, array)
    return elements[np.invert(is_in)]


def printv(string, verbose, **kwargs):
    """Print string if verbose is True

    parameters
    ----------
    string : str
    verbose : bool
    kwargs
        any kwargs for print()
    """
    if verbose:
        print(string, **kwargs)
