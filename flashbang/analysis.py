import numpy as np
import sys
from scipy.interpolate import interp1d
import astropy.units as units

g_to_msun = units.g.to(units.M_sun)


def extract_all_mass_tracers():
    """

    """
    pass


def extract_chk_mass_tracers(mass_grid, profile, params):
    """Return interpolated mass tracers for given profile quantities

    Returns: np.ndarray
        2D array of shape [n_tracers, n_params], where n_tracers=len(mass_grid)
        and n_params=len(params).

    parameters
    ----------
    mass_grid : [float]
        1D array of mass shells to track.
    profile : pd.DataFrame
        profile table from a single chk, with columns of params (including mass shells)
        and rows of radial zones (see: load_save.extract_profile).
    params : [str]
        list of profile quantities to extract.
    """
    n_tracers = len(mass_grid)
    n_params = len(params)
    out_array = np.zeros([n_tracers, n_params])

    for i, par in enumerate(params):
        func = interp1d(profile['mass'] * g_to_msun, profile[par])
        out_array[:, i] = func(mass_grid)

    return out_array
