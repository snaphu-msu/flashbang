import numpy as np
from astropy import units

"""
Module for calculating physical quantities
"""

g_to_msun = units.g.to(units.M_sun)


def get_mass_interior(radius, density):
    """Return interior (i.e. enclosed) mass for given radius/density profile

    Note
    ----
    interior mass at first radial point will be zero

    parameters
    ----------
    radius : np.ndarray
        1D array of radius values
    density : np.ndarray
        1D array of density values
    """
    n_points = len(radius)
    mass = np.zeros(n_points)
    mass_between = get_mass_between(radius=radius, density=density)

    for i, dm in enumerate(mass_between):
        mass[i+1] = mass[i] + dm

    return mass


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
    # TODO: This is not quite right:
    #           - Calculates volume *between* cell centres
    #           - Averages density at point halfway between cell centres
    #           - need to account for each cell size (do by blocks?)
    if len(radius) != len(density):
        raise ValueError('radius and density arrays are not the same length ' 
                         f'({len(radius)} and {len(radius)})')

    volume = 4/3 * np.pi * radius**3
    dv = volume[1:] - volume[:-1]
    avg_dens = 0.5 * (density[1:] + density[:-1])

    return dv * avg_dens * g_to_msun
