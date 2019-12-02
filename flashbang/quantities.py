import numpy as np
from astropy import units

"""
Module for calculating physical quantities
"""


def get_mass_between(radius, density):
    """Return mass contained between each point of given radius/density profile,
    using trapezoidal rule

    Note
    -----
    Assumes radius and density have same length units

    parameters
    ----------
    radius : np.ndarray
        1D array of radius values
    density : np.ndarray
        1D array of density values
    """
    if len(radius) != len(density):
        raise ValueError('radius and density arrays are not the same length ' 
                         f'({len(radius)} and {len(radius)})')

    volume = 4/3 * np.pi * radius**3
    dv = volume[1:] - volume[:-1]
    avg_dens = 0.5 * (density[1:] + density[:-1])

    return dv * avg_dens
