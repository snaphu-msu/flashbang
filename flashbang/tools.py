import numpy as np

"""
Misc. functions for general use
"""


def ensure_sequence(x):
    """Ensures given object is in the form of a sequence

    If object is scalar, returns as length-1 list
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    else:
        return [x, ]


