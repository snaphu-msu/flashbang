import numpy as np
import matplotlib.pyplot as plt


"""
General functions for plotting
"""


def setup_subplots(n_sub, max_cols=2, sub_figsize=(6, 5), **kwargs):
    """Constructs fig for given number of subplots

    returns : fig, ax

    parameters
    ----------
    n_sub : int
        number of subplots (axes)
    max_cols : int
        maximum number of columns to arange subplots
    sub_figsize : tuple
        figsize of each subplot
    **kwargs :
        args to be parsed to plt.subplots()
    """
    n_rows = int(np.ceil(n_sub / max_cols))
    n_cols = {False: 1, True: max_cols}.get(n_sub > 1)
    figsize = (n_cols*sub_figsize[0], n_rows*sub_figsize[1])
    return plt.subplots(n_rows, n_cols, figsize=figsize, **kwargs)

