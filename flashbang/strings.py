"""Functions for formatted strings
"""


def printv(string, verbose, **kwargs):
    """Print string if verbose is True
    """
    if verbose:
        print(string, **kwargs)
