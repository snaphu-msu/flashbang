"""Functions for loading/saving/reducing flash data

General terminology
-------------------
    dat: Integrated time-series quantities found in [model].dat file
    chk: Checkpoint data found in 'chk' files
    profile: Radial profile data as extracted from chk files
    multiprofile: A single Dataset of multiple profiles
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
import h5py

# flashbang
from .strings import printv
from . import paths
from . import quantities
from . import extract_tracers
from . import tools

# TODO:
#   - merge "get" functions into common class/function structure?
#   - multithread extract_timesteps
#   - function to extract colnames


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


# ===============================================================
#                      Cache files
# ===============================================================
def load_cache(name, run, model, model_set, chk=-1, verbose=True):
    """Load pre-cached data

    parameters
    ----------
    name : str
    run : str
    model : str
    model_set : str
    chk : int
    verbose : bool
    """
    filepath = paths.cache_filepath(name, run=run, model=model,
                                    model_set=model_set, chk=chk)

    printv(f'Loading {name} cache: {filepath}', verbose)

    if name in ['dat', 'chk_table']:
        data = pd.read_pickle(filepath)

    return data


# =======================================================================
#                      Dat files
# =======================================================================
def get_dat(run, model, model_set, cols_dict,
            reload=False, save=True, verbose=True):
    """Get reduced set of integrated quantities, as contained in [run].dat file

    Returns : pandas.DataFrame

    parameters
    ----------
    run: str
    model : str
    model_set : str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    reload : bool
    save : bool
    verbose : bool
    """
    dat_table = None

    # attempt to load cache file
    if not reload:
        try:
            dat_table = load_cache('dat', run=run, model=model,
                                   model_set=model_set, verbose=verbose)
        except FileNotFoundError:
            printv('dat cache not found, reloading', verbose)

    # fall back on loading raw .dat
    if dat_table is None:
        dat_table = extract_dat(run=run, model=model, model_set=model_set,
                                cols_dict=cols_dict, verbose=verbose)
        if save:
            save_dat_cache(dat_table, run=run, model=model,
                           model_set=model_set, verbose=verbose)

    return dat_table


def extract_dat(run, model, model_set, cols_dict, verbose=True):
    """Extract and reduce data from .dat file

    Returns : dict of 1D quantities

    parameters
    ----------
    run: str
    model : str
    model_set : str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    verbose : bool
    """
    filepath = paths.flash_filepath('dat', run=run, model=model, model_set=model_set)
    printv(f'Extracting dat: {filepath}', verbose=verbose)

    idxs = []
    keys = []

    for key, idx_1 in cols_dict.items():
        idxs += [idx_1 - 1]  # change to zero-indexed
        keys += [key]

    return pd.read_csv(filepath, usecols=idxs, names=keys, skiprows=1, header=None,
                       delim_whitespace=True, low_memory=False)


def save_dat_cache(dat, run, model, model_set,
                   verbose=True):
    """Save pre-extracted .dat quantities, for faster loading

    parameters
    ----------
    dat : pd.DataFrame
        data table as returned by extract_dat()
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    ensure_cache_dir_exists(model, model_set=model_set, verbose=False)
    filepath = paths.cache_filepath('dat', run=run, model=model, model_set=model_set)

    printv(f'Saving dat cache: {filepath}', verbose)
    dat.to_pickle(filepath)


def print_dat_colnames(run, model, model_set):
    """Print all column names from .dat file

    parameters
    ----------
    run : str
    model : str
    model_set : str
    """
    filepath = paths.flash_filepath('dat', run=run, model=model, model_set=model_set)

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
def get_multiprofile(run, model, model_set, chk_list=None, params=None,
                     derived_params=None, config=None, reload=False,
                     save=True, verbose=True):
    """Get all available profiles as multiprofile Dataset
        see: get_all_profiles()

    parameters
    ----------
    run : str
    model : str
    model_set : str
    chk_list : [int]
    params : [str]
    derived_params : [str]
    config : str
    reload : bool
    save : bool
    verbose : bool
    """
    def save_cache():
        if save:
            save_multiprofile_cache(multiprofile, run=run, model=model,
                                    model_set=model_set, verbose=verbose)

    if chk_list is None:
        chk_list = find_chk(run=run, model=model, model_set=model_set)

    # 1. Try loading multiprofile
    multiprofile = None
    if not reload:
        multiprofile = try_load_multiprofile(run=run, model=model,
                                             model_set=model_set,
                                             verbose=verbose)

    # 2. Reload individual profiles
    if multiprofile is None:
        profiles = get_all_profiles(run=run, model=model, model_set=model_set,
                                    chk_list=chk_list, params=params,
                                    derived_params=derived_params, save=save,
                                    verbose=verbose, config=config)

        multiprofile = join_profiles(profiles, verbose=verbose)
        save_cache()

    # 3. Check for missing profiles
    else:
        multi_chk = multiprofile.coords['chk'].values
        missing_chk = tools.get_missing_elements(chk_list, multi_chk)

        if len(missing_chk) > 0:
            printv('Loading missing profiles', verbose=verbose)
            missing_profiles = get_all_profiles(run=run, model=model,
                                                model_set=model_set,
                                                chk_list=missing_chk, params=params,
                                                save=save, verbose=verbose,
                                                derived_params=derived_params,
                                                config=config)

            multiprofile = append_to_multiprofile(multiprofile,
                                                  profiles=missing_profiles)
            save_cache()

    return multiprofile


def get_all_profiles(run, model, model_set, chk_list=None,
                     params=None, derived_params=None, config=None,
                     reload=False, save=True, verbose=True):
    """Get all available chk profiles
        see: get_profile()

    Returns: {chk: profile}

    parameters
    ----------
    run : str
    model : str
    model_set : str
    chk_list : [int]
    params : [str]
    derived_params : [str]
    config : str
    reload : bool
    save : bool
    verbose : bool
    """
    printv(f'Loading chk profiles', verbose=verbose)

    if chk_list is None:
        chk_list = find_chk(run=run, model=model, model_set=model_set)

    profiles = {}
    chk_max = chk_list[-1]

    for chk in chk_list:
        printv(f'\rchk: {chk}/{chk_max}', end='', verbose=verbose)

        profiles[chk] = get_profile(chk, run=run, model=model, model_set=model_set,
                                    params=params, derived_params=derived_params,
                                    config=config, reload=reload, save=save,
                                    verbose=False)
    printv('', verbose=verbose)
    return profiles


def try_load_multiprofile(run, model, model_set, verbose=True):
    """Attempt to load cached multiprofile

   Returns : xr.Dataset, or None

   parameters
   ----------
   run : str
   model : str
   model_set : str
   verbose : bool
   """
    multiprofile = None
    try:
        multiprofile = load_multiprofile_cache(run=run, model=model, model_set=model_set,
                                               verbose=verbose)
    except FileNotFoundError:
        printv('multiprofile cache not found, reloading', verbose=verbose)
        pass

    return multiprofile


def get_profile(chk, run, model, model_set, params=None,
                derived_params=None, config=None,
                reload=False, save=True, verbose=True):
    """Get reduced radial profile, as contained in checkpoint file
    Loads pre-extracted profile if available, otherwise from raw file

    Returns : dictionary of 1D arrays

    parameters
    ----------
    chk : int
    run : str
    model : str
    model_set : str
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

    # attempt to load cache file
    if not reload:
        try:
            profile = load_profile_cache(chk, run=run, model=model,
                                         model_set=model_set, verbose=verbose)
        except FileNotFoundError:
            printv('profile cache not found, reloading', verbose)

    # fall back on loading raw chk
    if profile is None:
        profile = extract_profile(chk, run=run, model=model, model_set=model_set,
                                  config=config, params=params,
                                  derived_params=derived_params)
        if save:
            save_profile_cache(profile, chk=chk, run=run, model=model,
                               model_set=model_set, verbose=verbose)

    return profile


def join_profiles(profiles, verbose=True):
    """Join multiple profile Datasets into a single Dataset (a 'multiprofile')

    Returns : xr.Dataset

    parameters
    ----------
    profiles : {chk: profile}
        dict of profile Datasets to join (with corresponding chk as keys)
    verbose : bool
    """
    printv('Joining profiles', verbose=verbose)

    joined = xr.concat(profiles.values(), dim='chk')
    joined.coords['chk'] = list(profiles.keys())

    return joined


def append_to_multiprofile(multiprofile, profiles, verbose=True):
    """
    Append new profiles to an existing multiprofile Dataset

    Returns : xr.Dataset

    parameters
    ----------
    multiprofile : xr.Dataset
        multiprofile to append onto
    profiles : {chk: profile}
        new profile Datasets to append, with chks as keys
    verbose : bool
    """
    # TODO: check no overlap? use merge?
    printv('Appending new profiles onto multiprofile', verbose=verbose)

    new_profiles = join_profiles(profiles, verbose=False)
    joined = xr.concat([multiprofile, new_profiles], dim='chk')

    return joined


def extract_profile(chk, run, model, model_set, params=None, derived_params=None,
                    config=None, verbose=True):
    """Extract and reduce profile data from chk file

    Returns : xr.Dataset

    parameters
    ----------
    chk : int
    run : str
    model : str
    model_set : str
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

    profile = xr.Dataset()
    chk_raw = load_chk(chk=chk, run=run, model=model, model_set=model_set)
    chk_data = chk_raw.all_data()

    for var in params:
        profile[var.strip()] = ('zone', np.array(chk_data[var]))

    if 'mass' in derived_params:
        chk_h5py = load_chk(chk=chk, run=run, model=model, model_set=model_set,
                            use_h5py=True)
        add_mass_profile(profile=profile, chk_h5py=chk_h5py)
        chk_h5py.close()

    n_zones = len(profile['zone'])
    profile.coords['zone'] = np.arange(n_zones)  # set coords (mostly for concat later)

    return profile


def add_mass_profile(profile, chk_h5py):
    """Calculate interior/enclosed mass profile, and adds to given table

    parameters
    ----------
    profile : xr.Dataset
        table as returned by extract_profile()
    chk_h5py : h5py.File
    """
    if ('r' not in profile) or ('dens' not in profile):
        raise ValueError(f'Need radius and density columns (r, dens) to calculate mass')

    mass = quantities.get_mass_interior(radius=np.array(profile['r']),
                                        density=np.array(profile['dens']),
                                        chk_h5py=chk_h5py)
    profile['mass'] = ('zone', mass)


def save_multiprofile_cache(multiprofile, run, model, model_set, verbose=True):
    """Save multiprofile to file for faster loading

    parameters
    ----------
    multiprofile : xr.Dataset
            Dataset of multiple profiles
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    ensure_cache_dir_exists(model, model_set=model_set, verbose=False)
    filepath = paths.cache_filepath('multiprofile', run=run, model=model,
                                    model_set=model_set)

    printv(f'Saving multiprofile cache: {filepath}', verbose)
    multiprofile.to_netcdf(filepath)


def load_multiprofile_cache(run, model, model_set, verbose=True):
    """Load pre-extracted profile (see: save_profile_cache)

    parameters
    ----------
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    filepath = paths.cache_filepath('multiprofile', run=run, model=model,
                                    model_set=model_set)
    printv(f'Loading multiprofile cache: {filepath}', verbose)
    return xr.load_dataset(filepath)


def save_profile_cache(profile, chk, run, model, model_set, verbose=True):
    """Save profile to file for faster loading

    parameters
    ----------
    profile : xr.Dataset
            table of profile properties as returned by extract_profile()
    chk : int
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    ensure_cache_dir_exists(model, model_set=model_set, verbose=False)
    filepath = paths.cache_filepath('profile', chk=chk, run=run, model=model,
                                    model_set=model_set)

    printv(f'Saving profile cache: {filepath}', verbose)
    profile.to_netcdf(filepath)


def load_profile_cache(chk, run, model, model_set, verbose=True):
    """Load pre-extracted profile (see: save_profile_cache)

    parameters
    ----------
    chk : int
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    filepath = paths.cache_filepath('profile', chk=chk, run=run, model=model,
                                    model_set=model_set)
    printv(f'Loading profile cache: {filepath}', verbose)
    return xr.load_dataset(filepath)


# ===============================================================
#                      Chk files
# ===============================================================
def find_chk(run, model, model_set, n_digits=4, verbose=True):
    """Return list of checkpoint (chk) files available in given directory
        returns as nparray of checkpoint numbers

    parameters
    ----------
    run : str
    model : str
    model_set : str
    n_digits : int
        number of digits at end of filename corresponding to checkpoint ID
    verbose : bool
    """
    output_path = paths.output_path(model=model, model_set=model_set)
    file_list = os.listdir(output_path)
    match_str = f'{run}_hdf5_chk_'
    chks = []
    printv(f'Searching for chk files: {output_path}/{match_str}' + n_digits*'*', verbose)

    for file in file_list:
        if match_str in file:
            chks += [int(file[-n_digits:])]

    return np.sort(chks)


def load_chk(chk, run, model, model_set, use_h5py=False):
    """Load checkpoint file using yt

    parameters
    ----------
    chk : int
    run : str
    model : str
    model_set : str
    use_h5py : bool
    """
    filepath = paths.flash_filepath('chk', chk=chk, run=run,
                                    model=model, model_set=model_set)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f'checkpoint {chk:04d} file does not exist: {filepath}')

    if use_h5py:
        return h5py.File(filepath, 'r')
    else:
        return yt.load(filepath)


# ===============================================================
#                      chk_table
# ===============================================================
def get_chk_table(run, model, model_set, reload=False, save=True, verbose=True):
    """Get table of scalar chk properties

    Returns: pd.DataFrame

    parameters
    ----------
    run : str
    model : str
    model_set : str
    reload : bool
    save : bool
    verbose : bool
    """
    chk_table = pd.DataFrame()

    if not reload:
        try:
            chk_table = load_cache('chk_table', run=run, model=model,
                                   model_set=model_set, verbose=verbose)
        except FileNotFoundError:
            printv('chk_table cache not found, reloading', verbose)

    if chk_table.empty:
        chk_table['chk'] = find_chk(run=run, model=model, model_set=model_set)
        chk_table.set_index('chk', inplace=True)

        if save:
            save_chk_table_cache(chk_table, run=run, model=model,
                                 model_set=model_set, verbose=verbose)

    return chk_table


def save_chk_table_cache(chk_table, run, model, model_set, verbose=True):
    """Saves pre-extracted chk_table to file

    parameters
    ----------
    chk_table : pd.Dataframe
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    ensure_cache_dir_exists(model, model_set=model_set, verbose=False)
    filepath = paths.cache_filepath('chk_table', run=run, model=model, model_set=model_set)
    printv(f'Saving chk_table cache: {filepath}', verbose)
    chk_table.to_pickle(filepath)


# ===============================================================
#                      Timesteps
# ===============================================================
# TODO:
#   - faster method? e.g. reading from .dat somehow?
#       - or reading chk without loading fully
#       - incorporating into multiprofile
# ===============================================================
def get_timesteps(run, model, model_set, params=('time', 'nstep'),
                  reload=False, save=True, verbose=True):
    """Get table of timestep quantities (time, n_steps, etc.) from chk files

    Returns : pandas.DataFrame

    parameters
    ----------
    run: str
    model : str
    model_set : str
    params : [str]
    reload : bool
    save : bool
    verbose : bool
    """
    timesteps = None

    # attempt to load cache file
    if not reload:
        try:
            timesteps = load_timesteps_cache(run=run, model=model, model_set=model_set,
                                             verbose=verbose)
        except FileNotFoundError:
            printv('timesteps cache not found, reloading', verbose)

    # fall back on loading from raw chk files
    if timesteps is None:
        chk_list = find_chk(run=run, model=model, model_set=model_set)

        timesteps = extract_timesteps(chk_list, run=run, model=model,
                                      model_set=model_set, params=params)

        if save:
            save_timesteps_cache(timesteps, run=run, model=model, model_set=model_set,
                                 verbose=verbose)

    return timesteps


def extract_timesteps(chk_list, run, model, model_set, params=('time', 'nstep')):
    """Extract timestep quantities from chk files

    Returns: pd.DataFrame()

    parameters
    ----------
    chk_list : [int]
    run : str
    model : str
    model_set : str
    params : [str]
    """
    t0 = time.time()
    arrays = dict.fromkeys(params)
    chk0 = load_chk(chk_list[0], run=run, model=model, model_set=model_set)

    for par in params:
        par_type = type(chk0.parameters[par])
        arrays[par] = np.zeros_like(chk_list, dtype=par_type)

    for i, chk in enumerate(chk_list[1:]):
        chk_raw = load_chk(chk, run=run, model=model, model_set=model_set)
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


def save_timesteps_cache(timesteps, run, model, model_set, verbose=True):
    """Save pre-extracted chk timesteps to file

    parameters
    ----------
    timesteps : pd.DataFrame
        table of chk timesteps, as returned by extract_timesteps()
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    ensure_cache_dir_exists(model, model_set=model_set, verbose=False)
    filepath = paths.cache_filepath('timesteps', run=run, model=model, model_set=model_set)

    printv(f'Saving timesteps cache: {filepath}', verbose)
    timesteps_out = timesteps.reset_index()
    timesteps_out.to_pickle(filepath)


def load_timesteps_cache(run, model, model_set, verbose=True):
    """Load pre-extracted chk timesteps to file

    parameters
    ----------
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    filepath = paths.cache_filepath('timesteps', run=run, model=model, model_set=model_set)
    printv(f'Loading timesteps cache: {filepath}', verbose)

    timesteps = pd.read_pickle(filepath)
    timesteps.set_index('chk', inplace=True)
    return timesteps


# ===============================================================
#                      Log files
# ===============================================================
def get_bounce_time(run, model, model_set, match_str='Bounce', verbose=True):
    """Get bounce time (s) from .log file

    parameters
    ----------
    run : str
    model : str
    model_set : str
    match_str : str
        String which immediately precedes the bounce time
    verbose : bool
    """
    filepath = paths.flash_filepath('log', run=run, model=model, model_set=model_set)
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
def get_tracers(run, model, model_set, profiles=None, params=None, mass_grid=None,
                reload=False, save=True, config=None, verbose=True):
    """Get mass tracers from interpolated chk profiles

    Returns : xr.Dataset

    parameters
    ----------
    run: str
    model : str
    model_set : str
    profiles : xr.Dataset
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
            tracers = load_tracers_cache(run=run, model=model,
                                         model_set=model_set,
                                         verbose=verbose)
        except FileNotFoundError:
            printv('tracers cache not found, reloading', verbose)

    # fall back on re-extracting
    if tracers is None:
        c = load_config(config, verbose=verbose)

        if mass_grid is None:
            mass_def = c['tracers']['mass_grid']
            mass_grid = np.linspace(mass_def[0], mass_def[1], mass_def[2])

        if params is None:
            params = c['tracers']['params']

        if profiles is None:
            chk_list = find_chk(run=run, model=model, model_set=model_set)

            profiles = get_multiprofile(run=run, model=model,
                                        model_set=model_set, chk_list=chk_list,
                                        params=params, verbose=verbose)

        tracers = extract_tracers.extract_multi_tracers(mass_grid,
                                                        profiles=profiles,
                                                        params=params,
                                                        verbose=verbose)
        if save:
            save_tracers_cache(tracers, run=run, model=model,
                               model_set=model_set, verbose=verbose)
    return tracers


def save_tracers_cache(tracers, run, model, model_set, verbose=True):
    """Save pre-extracted mass tracers to file

    parameters
    ----------
    tracers : xr.Dataset
        mass tracer data, as returned by analysis.extract_multi_tracers()
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    filepath = paths.cache_filepath('tracers', run=run, model=model, model_set=model_set)
    printv(f'Saving tracers cache: {filepath}', verbose)
    tracers.to_netcdf(filepath)


def load_tracers_cache(run, model, model_set, verbose=True):
    """Load pre-extracted mass tracers from file

    parameters
    ----------
    run : str
    model : str
    model_set : str
    verbose : bool
    """
    filepath = paths.cache_filepath('tracers', run=run, model=model, model_set=model_set)
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


def ensure_cache_dir_exists(model, model_set, verbose=True):
    """Create cache directory if it doesn't exist

    parameters
    ----------
    model : str
    model_set : str
    verbose : bool
    """
    path = paths.model_cache_path(model, model_set=model_set)
    try_mkdir(path, skip=True, verbose=verbose)
