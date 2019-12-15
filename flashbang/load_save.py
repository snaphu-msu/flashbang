"""Functions for loading/saving/reducing flash data

General terminology
-------------------
    dat: Integrated time-series quantities found in [model].dat file
    chk: Checkpoint data found in 'chk' files
    profile: Radial profile data as extracted from chk files
    log: Data printed to terminal during model, stored in .log files
    tracers: Mass shell tracers, interpolated from profiles

    extract: Extract and reduce data from raw output files
    save: Save pre-extracted data to file
    load: Load pre-extracted data from file
    get: Get reduced data by first attempting 'load', then fall back on 'extract'
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import configparser
import ast
import subprocess
import sys
import yt
import time

# flashbang
from .strings import printv
from . import paths
from . import quantities
from . import analysis

# TODO:
#   - merge "get" functions into common class/function structure?
#   - Refactor profiles into large xarray?
#   - multithread extract_timesteps
#   - function to extract colnames
#   - rename 'reload' to 'redo' or similar (avoid confusion with 'load')?


# =======================================================================
#                      Config files
# =======================================================================
def load_config(name=None, verbose=True):
    """Load .ini config file and return as dict

    parameters
    ----------
    name: str
        label of config file to load
    verbose : bool
    """
    filepath = paths.config_filepath(name=name)
    printv(f'Loading config: {filepath}', verbose)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Config file not found: {filepath}')

    ini = configparser.ConfigParser()
    ini.read(filepath)

    config = {}
    for section in ini.sections():
        config[section] = {}
        for option in ini.options(section):
            config[section][option] = ast.literal_eval(ini.get(section, option))

    return config


# =======================================================================
#                      Dat files
# =======================================================================
def get_dat(model, cols_dict, run='run', reload=False, save=True, verbose=True):
    """Get reduced set of integrated quantities, as contained in [run].dat file

    Returns : pandas.DataFrame

    parameters
    ----------
    model : str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    run: str
    reload : bool
    save : bool
    verbose : bool
    """
    dat_table = None

    # attempt to load temp file
    if not reload:
        try:
            dat_table = load_dat_cache(model=model, run=run, verbose=verbose)
        except FileNotFoundError:
            pass

    # fall back on loading raw .dat
    if dat_table is None:
        dat_table = extract_dat(model, cols_dict=cols_dict, run=run, verbose=verbose)
        if save:
            save_dat_cache(dat_table, model=model, run=run, 
                           verbose=verbose)

    return dat_table


def extract_dat(model, cols_dict, run='run', verbose=True):
    """Extract and reduce data from .dat file

    Returns : dict of 1D quantities

    parameters
    ----------
    model : str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    run: str
    verbose : bool
    """
    filepath = paths.dat_filepath(model=model, run=run)
    printv(f'Extracting dat: {filepath}', verbose=verbose)

    idxs = []
    keys = []

    for key, idx_1 in cols_dict.items():
        idxs += [idx_1 - 1]  # change to zero-indexed
        keys += [key]

    return pd.read_csv(filepath, usecols=idxs, names=keys, skiprows=1, header=None,
                       delim_whitespace=True, low_memory=False)


def save_dat_cache(dat, model, run='run', verbose=True):
    """Save pre-extracted .dat quantities, for faster loading

    parameters
    ----------
    dat : pd.DataFrame
        data table as returned by extract_dat()
    model : str
    run : str
    verbose : bool
    """
    ensure_temp_dir_exists(model, verbose=False)
    filepath = paths.dat_temp_filepath(model=model, run=run)

    printv(f'Saving dat cache: {filepath}', verbose)
    dat.to_pickle(filepath)


def load_dat_cache(model, run='run', verbose=True):
    """Load pre-extracted .dat quantities (see: save_dat_cache)

    parameters
    ----------
    model : str
    run : str
    verbose : bool
    """
    filepath = paths.dat_temp_filepath(model=model, run=run)
    printv(f'Loading dat cache: {filepath}', verbose)
    return pd.read_pickle(filepath)


def print_dat_colnames(model, run='run'):
    """Print all column names from .dat file

    parameters
    ----------
    model : str
    run : str
    """
    filepath = paths.dat_filepath(run=run, model=model)

    with open(filepath, 'r') as f:
        colnames = f.readline().split()

    count = 1
    for word in colnames:
        if str(count) in word:
            print(f'\n{count}', end=' ')
            count += 1
        else:
            print(word, end=' ')
            
            
# ===============================================================
#                      Profiles
# ===============================================================
def get_multi_profiles(model, run='run', chk_list=None, params=None, derived_params=None,
                       reload=False, save=True, verbose=True):
    """Get all available chk profiles
        see: get_profile()

    parameters
    ----------
    model : str
    run : str
    chk_list : [int]
    params : [str]
    derived_params : [str]
    reload : bool
    save : bool
    verbose : bool
    """
    printv(f'Loading chk profiles', verbose=verbose)

    if chk_list is None:
        chk_list = find_chk(model=model, match_str=f'{run}_hdf5_chk_')

    profiles = {}
    chk_max = chk_list[-1]

    for chk in chk_list:
        printv(f'\rchk: {chk}/{chk_max}', end='', verbose=verbose)
        profiles[chk] = get_profile(chk, model=model, run=run, params=params,
                                    derived_params=derived_params,
                                    reload=reload, save=save, verbose=False)
    printv('', verbose=verbose)
    return profiles


def get_profile(chk, model, run='run', params=None, derived_params=None, config=None,
                reload=False, save=True, verbose=True):
    """Get reduced radial profile, as contained in checkpoint file
    Loads pre-extracted profile if available, otherwise from raw file

    Returns : dictionary of 1D arrays

    parameters
    ----------
    chk : int
    model : str
    run : str
    params : [str]
        profile parameters to extract and return from chk file
    derived_params : [str]
        secondary profile parameters, derived from primary parameters
    config : str
    reload : bool
        force reload from chk file, else try to load pre-extracted profile
    save : bool
        save extracted profile to file for faster loading
    verbose : bool
    """
    profile = None

    # attempt to load temp file
    if not reload:
        try:
            profile = load_profile_cache(chk, model=model, run=run, verbose=verbose)
        except FileNotFoundError:
            pass

    # fall back on loading raw chk
    if profile is None:
        profile = extract_profile(chk, model=model, run=run, config=config,
                                  params=params, derived_params=derived_params)
        if save:
            save_profile_cache(profile, chk=chk, model=model, run=run, verbose=verbose)

    return profile


def extract_profile(chk, model, run='run', params=None, derived_params=None,
                    config=None, verbose=True):
    """Extract and reduce profile data from chk file

    Returns : pd.DataFrame

    parameters
    ----------
    chk : int
    model : str
    run : str
    params : [str]
        profile parameters to extract and return from chk file
    derived_params : [str]
        secondary profile parameters, derived from primary parameters
    config : str
    verbose : bool
    """
    if params is None:
        c = load_config(config, verbose=verbose)
        params = c['profiles']['params'] + c['profiles']['isotopes']
        derived_params = c['profiles']['derived_params']

    profile = pd.DataFrame()
    chk_raw = load_chk(chk=chk, model=model, run=run)
    chk_data = chk_raw.all_data()

    for var in params:
        profile[var.strip()] = np.array(chk_data[var])

    if 'mass' in derived_params:
        add_mass_profile(profile)

    return profile


def add_mass_profile(profile):
    """Calculate interior/enclosed mass profile, and adds to given table

    parameters
    ----------
    profile : pd.DataFrame
        table as returned by extract_profile()
    """
    if ('r' not in profile.columns) or ('dens' not in profile.columns):
        raise ValueError(f'Need radius and density columns (r, dens) to calculate mass')

    profile['mass'] = quantities.get_mass_interior(radius=np.array(profile['r']),
                                                   density=np.array(profile['dens']))


def save_profile_cache(profile, chk, model, run='run', verbose=True):
    """Save profile to file for faster loading

    parameters
    ----------
    profile : pd.DataFrame
            table of profile properties as returned by extract_profile()
    chk : int
    model : str
    run : str
    verbose : bool
    """
    ensure_temp_dir_exists(model, verbose=False)
    filepath = paths.profile_filepath(chk=chk, model=model, run=run)

    printv(f'Saving profile cache: {filepath}', verbose)
    profile.to_pickle(filepath)


def load_profile_cache(chk, model, run='run', verbose=True):
    """Load pre-extracted profile (see: save_profile_cache)

    parameters
    ----------
    chk : int
    model : str
    run : str
    verbose : bool
    """
    filepath = paths.profile_filepath(chk=chk, model=model, run=run)
    printv(f'Loading profile cache: {filepath}', verbose)
    return pd.read_pickle(filepath)


# ===============================================================
#                      Chk files
# ===============================================================
def find_chk(model, match_str='hdf5_chk_', n_digits=4):
    """Return list of checkpoint (chk) files available in given directory
        returns as nparray of checkpoint numbers

    parameters
    ----------
    model : str
    match_str : str
        string common to all chk filenames
    n_digits : int
        number of digits at end of filename corresponding to checkpoint ID
    """
    output_path = paths.output_path(model=model)
    file_list = os.listdir(output_path)
    chks = []

    for file in file_list:
        if match_str in file:
            chks += [int(file[-n_digits:])]

    return np.sort(chks)


def load_chk(chk, model, run='run'):
    """Load checkpoint file using yt

    parameters
    ----------
    chk : int
    model : str
    run : str
    """
    filepath = paths.chk_filepath(chk=chk, model=model, run=run)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f'checkpoint {chk:04d} file does not exist: {filepath}')

    return yt.load(filepath)


# ===============================================================
#                      Timesteps
# ===============================================================
# TODO:
#   - faster method? e.g. reading from .dat somehow?
#       - or reading chk without loading fully
#       - incorporating into xr.dataset metadata
# ===============================================================
def get_timesteps(model, run='run', params=('time', 'nstep'),
                  reload=False, save=True, verbose=True):
    """Get table of timestep quantities (time, n_steps, etc.) from chk files

    Returns : pandas.DataFrame

    parameters
    ----------
    model : str
    run: str
    params : [str]
    reload : bool
    save : bool
    verbose : bool
    """
    timesteps = None

    # attempt to load temp file
    if not reload:
        try:
            timesteps = load_timesteps_cache(model=model, run=run, verbose=verbose)
        except FileNotFoundError:
            pass

    # fall back on loading from raw chk files
    if timesteps is None:
        chk_list = find_chk(model=model, match_str=f'{run}_hdf5_chk_')
        timesteps = extract_timesteps(chk_list, model, run=run, params=params)

        if save:
            save_timesteps_cache(timesteps, model=model, run=run, verbose=verbose)

    return timesteps


def extract_timesteps(chk_list, model, run='run', params=('time', 'nstep')):
    """Extract timestep quantities from chk files

    Returns: pd.DataFrame()

    parameters
    ----------
    chk_list : [int]
    model : str
    run : str
    params : [str]
    """
    t0 = time.time()
    arrays = dict.fromkeys(params)
    chk0 = load_chk(chk_list[0], model=model, run=run)

    for par in params:
        par_type = type(chk0.parameters[par])
        arrays[par] = np.zeros_like(chk_list, dtype=par_type)

    for i, chk in enumerate(chk_list[1:]):
        chk_raw = load_chk(chk, model=model, run=run)
        for par in params:
            arrays[par][i+1] = chk_raw.parameters[par]

    timesteps = pd.DataFrame()
    timesteps['chk'] = chk_list

    for par, arr in arrays.items():
        timesteps[par] = arr

    timesteps.set_index('chk', inplace=True)

    t1 = time.time()
    print('='*20, f'\nTotal time: {t1-t0:.3f} s\n')
    return timesteps


def save_timesteps_cache(timesteps, model, run='run', verbose=True):
    """Save pre-extracted chk timesteps to file

    parameters
    ----------
    timesteps : pd.DataFrame
        table of chk timesteps, as returned by extract_timesteps()
    model : str
    run : str
    verbose : bool
    """
    ensure_temp_dir_exists(model, verbose=False)
    filepath = paths.timesteps_filepath(model, run=run)

    printv(f'Saving timesteps cache: {filepath}', verbose)
    timesteps_out = timesteps.reset_index()
    timesteps_out.to_pickle(filepath)


def load_timesteps_cache(model, run='run', verbose=True):
    """Load pre-extracted chk timesteps to file

    parameters
    ----------
    model : str
    run : str
    verbose : bool
    """
    filepath = paths.timesteps_filepath(model=model, run=run)
    printv(f'Loading timesteps cache: {filepath}', verbose)

    timesteps = pd.read_pickle(filepath)
    timesteps.set_index('chk', inplace=True)
    return timesteps


# ===============================================================
#                      Log files
# ===============================================================
def get_bounce_time(model, run='run', match_str='Bounce', verbose=True):
    """Get bounce time (s) from .log file

    parameters
    ----------
    model : str
    run : str
    match_str : str
        String which immediately precedes the bounce time
    verbose : bool
    """
    filepath = paths.log_filepath(run=run, model=model)
    bounce_time = 0.0
    printv(f'Getting bounce time: {filepath}', verbose)

    with open(filepath, 'r') as f:
        for line in f:
            if match_str in line:
                terms = line.split()
                bounce_time = float(terms[1])
                printv(f'Bounce = {bounce_time:.4f} s', verbose)
                break

        if bounce_time == 0.0:
            printv('Bounce time not found! Returning 0.0 s', verbose)

    return bounce_time


# ===============================================================
#                      Mass Tracers
# ===============================================================
def get_tracers(model, run='run', profiles=None, params=None, mass_grid=None,
                reload=False, save=True, config=None, verbose=True):
    """Get mass tracers from interpolated chk profiles

    Returns : xr.Dataset

    parameters
    ----------
    model : str
    run: str
    profiles : {pd.Dataframe}
    params : [str]
    mass_grid : [float]
    reload : bool
    save : bool
    config : str
    verbose : bool
    """
    tracers = None

    # attempt to load from cache
    if not reload:
        try:
            tracers = load_tracers_cache(model=model, run=run, verbose=verbose)
        except FileNotFoundError:
            pass

    # fall back on re-extracting
    if tracers is None:
        c = load_config(config, verbose=verbose)

        if mass_grid is None:
            mass_def = c['tracers']['mass_grid']
            mass_grid = analysis.get_mass_grid(mass_def[0], mass_def[1], mass_def[2])

        if params is None:
            params = c['tracers']['params']

        if profiles is None:
            chk_list = find_chk(model=model, match_str=f'{run}_hdf5_chk_')
            profiles = get_multi_profiles(model=model, run=run, chk_list=chk_list,
                                          params=params, verbose=verbose)

        tracers = analysis.extract_multi_tracers(mass_grid,
                                                 profiles=profiles,
                                                 params=params,
                                                 verbose=verbose)
        if save:
            save_tracers_cache(tracers, model=model, run=run, verbose=verbose)

    return tracers


def save_tracers_cache(tracers, model, run='run', verbose=True):
    """Save pre-extracted mass tracers to file

    parameters
    ----------
    tracers : xr.Dataset
        mass tracer data, as returned by analysis.extract_multi_tracers()
    model : str
    run : str
    verbose : bool
    """
    filepath = paths.tracers_filepath(model=model, run=run)
    printv(f'Saving tracers cache: {filepath}', verbose)
    tracers.to_netcdf(filepath)


def load_tracers_cache(model, run='run', verbose=True):
    """Load pre-extracted mass tracers from file

    parameters
    ----------
    model : str
    run : str
    verbose : bool
    """
    filepath = paths.tracers_filepath(model=model, run=run)
    printv(f'Loading tracers cache: {filepath}', verbose)
    return xr.load_dataset(filepath)


# ===============================================================
#              Misc. file things
# ===============================================================
def try_mkdir(path, skip=False, verbose=True):
    """Try to make given directory

    parameters
    ----------
    path: str
    skip : bool
        do nothing if directory already exists
        if skip=false, will ask to overwrite an existing directory
    verbose : bool
    """
    printv(f'Creating directory  {path}', verbose)
    if os.path.exists(path):
        if skip:
            printv('Directory already exists - skipping', verbose)
        else:
            print('Directory exists')
            cont = input('Overwrite (DESTROY)? (y/[n]): ')

            if cont == 'y' or cont == 'Y':
                subprocess.run(['rm', '-r', path])
                subprocess.run(['mkdir', path])
            elif cont == 'n' or cont == 'N':
                sys.exit()
    else:
        subprocess.run(['mkdir', '-p', path], check=True)


def ensure_temp_dir_exists(model, verbose=True):
    """Ensure temp directory exists (create if not)

    parameters
    ----------
    model : str
    verbose : bool
    """
    temp_path = paths.temp_path(model)
    try_mkdir(temp_path, skip=True, verbose=verbose)
