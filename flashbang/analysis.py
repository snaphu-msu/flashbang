import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import astropy.units as units

# flashbang
from .strings import printv

g_to_msun = units.g.to(units.M_sun)


def extract_multi_tracers(mass_grid, profiles, params, verbose=True):
    """Iterate over chk profiles and interpolate mass shell tracers for each

    Returns: xr.Dataset
        3D dataset of shape [n_profiles, n_tracers, n_params],
        where n_profiles=len(profiles), n_tracers=len(mass_grid) and n_params=len(params).

    parameters
    ----------
    mass_grid : [float]
        1D array of mass shells to track.
    profiles : xr.Dataset
        set of profile tables for each chk, with columns of params (including mass shells)
        and rows of radial zones (see: load_save.extract_profile).
    params : [str]
        list of profile quantities to extract.
    verbose : bool
    """
    printv(f'Extracting mass tracers from chk profiles', verbose=verbose)

    chk_list = profiles.coords['chk'].values
    data_cube = np.zeros([len(chk_list),
                          len(mass_grid),
                          len(params)])

    for i, chk in enumerate(chk_list):
        printv(f'\rchk: {chk}/{chk_list[-1]}', verbose, end='')

        data_cube[i, :, :] = extract_tracers(mass_grid=mass_grid,
                                             profile=profiles.sel(chk=i),
                                             params=params)
    # construct xarray Dataset
    tracers = xr.Dataset()
    tracers.coords['chk'] = chk_list
    tracers.coords['mass'] = mass_grid

    for i, par in enumerate(params):
        tracers[par] = (('chk', 'mass'), data_cube[:, :, i])

    printv('', verbose)
    return tracers


def extract_tracers(mass_grid, profile, params):
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
    out_array = np.zeros([len(mass_grid), len(params)])

    for i, par in enumerate(params):
        interp_func = interp1d(profile['mass'], profile[par])
        out_array[:, i] = interp_func(mass_grid)

    return out_array
