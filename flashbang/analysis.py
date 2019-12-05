import numpy as np
from scipy.interpolate import interp1d
import astropy.units as units

# flashbang
from .strings import printv

g_to_msun = units.g.to(units.M_sun)

# TODO:
#   - save tracer data cube
#   - load tracers


def extract_multi_mass_tracers(mass_grid, profiles, params, verbose=True):
    """Iterate over chk profiles and interpolate mass shell tracers for each

    Returns: np.ndarray
        3D array of shape [n_profiles, n_tracers, n_params],
        where n_profiles=len(profiles), n_tracers=len(mass_grid) and n_params=len(params).

    parameters
    ----------
    mass_grid : [float]
        1D array of mass shells to track.
    profiles : {pd.DataFrame}
        set of profile tables for each chk, with columns of params (including mass shells)
        and rows of radial zones (see: load_save.extract_profile).
    params : [str]
        list of profile quantities to extract.
    verbose : bool
    """
    printv(f'Extracting mass tracers from chk profiles', verbose=verbose)

    n_profiles = len(profiles)
    n_tracers = len(mass_grid)
    n_params = len(params)

    data_cube = np.zeros([n_profiles, n_tracers, n_params])

    for i, chk in enumerate(profiles.keys()):
        printv(f'\rchk: {chk} ({100 * i / n_profiles:.0f}%)', verbose, end='')
        profile = profiles[i]

        data_cube[i, :, :] = extract_mass_tracers(mass_grid=mass_grid,
                                                  profile=profile,
                                                  params=params)
    printv('', verbose)
    return data_cube


def extract_mass_tracers(mass_grid, profile, params):
    """Return interpolated mass tracers for given chk profile

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
