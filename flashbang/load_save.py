"""Functions for loading/saving/reducing flash data

General terminology
-------------------
    dat: Time-integrated data found in [model].dat file
    chk: Checkpoint data found in 'chk' files
    profile: Radial profile data as extracted from chk files

    extract: Extract and reduce data from raw output files
    save: Save pre-extracted data to file
    load: Load pre-extracted data from file
    get: Get reduced data by first attempting 'load', then fall back on 'extract'
"""
import os
import numpy as np
import pandas as pd
import configparser
import ast
import subprocess
import sys
import yt

# bangpy
from . import paths
from .strings import printv


def load_config(name='default', verbose=True):
    """Load .ini config file and return as dict

    parameters
    ----------
    name: str (optional)
        label of config file to load
    verbose : bool (optional)
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


def get_dat(model, cols_dict, run='run', runs_path=None, runs_prefix='run_',
            verbose=True, save=True, reload=False):
    """Get reduced set of integrated quantities, as contained in [run].dat file

    Returns : pandas.DataFrame

    parameters
    ----------
    model : str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    run: str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    save : bool (optional)
    reload : bool (optional)
    """
    dat_table = None

    # attempt to load temp file
    if not reload:
        try:
            dat_table = load_dat_cache(model=model, run=run, runs_path=runs_path,
                                       runs_prefix=runs_prefix, verbose=verbose)
        except FileNotFoundError:
            pass

    # fall back on loading raw .dat
    if dat_table is None:
        dat_table = extract_dat(model, cols_dict=cols_dict, run=run,
                                runs_path=runs_path, runs_prefix=runs_prefix)
        if save:
            save_dat_cache(dat_table, model=model, run=run, runs_path=runs_path,
                           runs_prefix=runs_prefix, verbose=verbose)

    return dat_table


def extract_dat(model, cols_dict, run='run', runs_path=None,
                runs_prefix='run_', verbose=True):
    """Extract and reduce data from .dat file

    Returns : dict of 1D quantities

    parameters
    ----------
    model : str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    run: str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    """
    filepath = paths.dat_filepath(model=model, run=run, runs_path=runs_path,
                                  runs_prefix=runs_prefix)

    printv(f'Extracting dat: {filepath}', verbose=verbose)

    idxs = []
    keys = []
    for key, idx_1 in cols_dict.items():
        idxs += [idx_1 - 1]  # change to zero-indexed
        keys += [key]

    return pd.read_csv(filepath, usecols=idxs, names=keys, skiprows=1, header=None,
                       delim_whitespace=True)


def save_dat_cache(dat, model, run='run', runs_path=None, runs_prefix='run_', verbose=True):
    """Save pre-extracted .dat quantities, for faster loading

    parameters
    ----------
    dat : pd.DataFrame
        table as returned by extract_dat()
    model : str
    run : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    """
    ensure_temp_dir_exists(model, runs_path=runs_path, runs_prefix=runs_prefix,
                           verbose=verbose)
    filepath = paths.dat_temp_filepath(model=model, run=run, runs_path=runs_path,
                                       runs_prefix=runs_prefix)

    printv(f'Saving dat cache: {filepath}', verbose)
    dat.to_feather(filepath)


def load_dat_cache(model, run='run', runs_path=None, runs_prefix='run_', verbose=True):
    """Load pre-extracted .dat quantities (see: save_dat_cache)

    parameters
    ----------
    model : str
    run : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    """
    filepath = paths.dat_temp_filepath(model=model, run=run, runs_path=runs_path,
                                       runs_prefix=runs_prefix)
    printv(f'Loading dat cache: {filepath}', verbose)
    if os.path.exists(filepath):
        return pd.read_feather(filepath)
    else:
        raise FileNotFoundError


def get_profile(chk, model, run='run', output_dir='output',
                runs_path=None, runs_prefix='run_', o_path=None,
                params=('temp', 'dens', 'pres'), reload=False,
                save=True, verbose=True):
    """Get reduced radial profile, as contained in checkpoint file
    Loads pre-extracted profile if available, otherwise from raw file

    Returns : dictionary of 1D arrays

    parameters
    ----------
    chk : int
    model : str
    run : str (optional)
    output_dir : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    o_path : str (optional)
    params : [] (optional)
        profile parameters to extract and return from chk file
    reload : bool (optional)
        force reload from chk file, else try to load pre-extracted profile
    save : bool
        save extracted profile to file for faster loading
    verbose : bool (optional)
    """
    profile = None

    # attempt to load temp file
    if not reload:
        try:
            profile = load_profile_cache(chk, model=model, run=run, runs_path=runs_path,
                                         runs_prefix=runs_prefix, verbose=verbose)
        except FileNotFoundError:
            pass

    # fall back on loading raw chk
    if profile is None:
        profile = extract_profile(chk, model=model, run=run, output_dir=output_dir,
                                  runs_path=runs_path, runs_prefix=runs_prefix,
                                  o_path=o_path, params=params)
        if save:
            save_profile_cache(profile, chk=chk, model=model, run=run, runs_path=runs_path,
                               runs_prefix=runs_prefix, verbose=verbose)

    return profile


def extract_profile(chk, model, run='run', output_dir='output',
                    runs_path=None, runs_prefix='run_', o_path=None,
                    params=('r', 'temp', 'dens', 'pres')):
    """Extract and reduce profile data from chk file

    Returns : pd.DataFrame

    parameters
    ----------
    chk : int
    model : str
    run : str (optional)
    output_dir : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    o_path : str (optional)
    params : [] (optional)
        profile parameters to extract and return from chk file
    """
    profile = pd.DataFrame()
    chk_raw = load_chk(chk=chk, model=model, run=run, output_dir=output_dir,
                       runs_path=runs_path, runs_prefix=runs_prefix, o_path=o_path)
    chk_data = chk_raw.all_data()

    for var in params:
        profile[var.strip()] = np.array(chk_data[var])

    return profile


def save_profile_cache(profile, chk, model, run='run', runs_path=None,
                       runs_prefix='run_', verbose=True):
    """Save profile to file for faster loading

    parameters
    ----------
    profile : pd.DataFrame
            table as returned by extract_profile()
    chk : int
    model : str
    run : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    """
    ensure_temp_dir_exists(model, runs_path=runs_path, runs_prefix=runs_prefix,
                           verbose=verbose)
    filepath = paths.profile_filepath(chk=chk, model=model, run=run,
                                      runs_path=runs_path, runs_prefix=runs_prefix)

    printv(f'Saving profile cache: {filepath}', verbose)
    profile.to_feather(filepath)


def load_profile_cache(chk, model, run='run', runs_path=None,
                       runs_prefix='run_', verbose=True):
    """Load pre-extracted profile (see: save_profile_cache)

    parameters
    ----------
    chk : int
    model : str
    run : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    """
    filepath = paths.profile_filepath(chk=chk, model=model, run=run,
                                      runs_path=runs_path, runs_prefix=runs_prefix)
    printv(f'Loading profile cache: {filepath}', verbose)

    if os.path.exists(filepath):
        return pd.read_feather(filepath)
    else:
        raise FileNotFoundError


def load_chk(chk, model, run='run', output_dir='output',
             runs_path=None, runs_prefix='run_', o_path=None):
    """Load checkpoint file for given model

    parameters
    ----------
    chk : int
    model : str
    run : str (optional)
    output_dir : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    o_path : str (optional)
    """
    filepath = paths.chk_filepath(chk=chk, model=model, run=run,
                                  output_dir=output_dir, runs_path=runs_path,
                                  runs_prefix=runs_prefix, o_path=o_path)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f'checkpoint {chk:04d} file does not exist: {filepath}')

    return yt.load(filepath)


def find_chk(path, match_str='hdf5_chk_', n_digits=4):
    """Return list of checkpoint (chk) files available in given directory
        returns as nparray of checkpoint numbers

    parameters
    ----------
    path : str
        path to directory to look in
    match_str : str
        string to match for in filename, to identify chk files
    n_digits : int
        number of digits at end of filename corresponding to checkpoint ID
    """
    file_list = os.listdir(path)
    chks = []

    for file in file_list:
        if match_str in file:
            chks += [int(file[-n_digits:])]

    return np.sort(chks)


def get_bounce_time(model, run='run', runs_path=None, runs_prefix='run_',
                    match_str='Bounce', verbose=True):
    """Return bounce time (s)

    Assumes the bounce time immediately follows 'match_str'
    """
    filepath = paths.log_filepath(run=run, model=model, runs_path=runs_path,
                                  runs_prefix=runs_prefix)
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


def reduce_snec_profile(profile_dict):
    """Reduce given profile dictionary into a 2D nparray
        Returns: profile_array, time, mass_grid

    parameters
    ----------
    profile_dict : {}
        Dictionary containing profile data, as returned from load_snec_xg()
    """
    time = np.array(list(profile_dict.keys()))
    n_time = len(time)
    
    mass_grid = profile_dict[time[0]][:, 0]
    n_mass = len(mass_grid)
    
    profile_array = np.zeros((n_time, n_mass))
    
    for i, key in enumerate(time):
        profile_array[i, :] = profile_dict[key][:, 1]
    
    return profile_array, time, mass_grid
    

def load_snec_xg(filepath, verbose=True):
    """Load mass tracers from SNEC output .xg file, returns as dict
    """    
    printv(f'Loading: {filepath}', verbose)
    n_lines = fast_line_count(filepath)

    profile = {}
    with open(filepath, 'r') as rf:
        count = 0
        for line in rf:
            # if verbose:
            sys.stdout.write(f'\r{100 * count/n_lines:.1f}%')
            cols = line.split()

            # Beginning of time data - make key for this time
            if 'Time' in line:
                time = float(cols[-1])
                profile[time] = []

            # In time data -- build x,y arrays
            elif len(cols) == 2:
                profile[time].append(np.fromstring(line, sep=' '))

            # End of time data (blank line) -- make list into array
            else:
                profile[time] = np.array(profile[time])
            count += 1

    if verbose:
        sys.stdout.write('\n')
    return profile


def print_dat_colnames(model, run='run', runs_path=None, runs_prefix='run_'):
    """Print all column names from .dat file
    """
    filepath = paths.dat_filepath(run=run, model=model, runs_prefix=runs_prefix,
                                  runs_path=runs_path)
    with open(filepath, 'r') as f:
        colnames = f.readline().split()

    count = 1
    for word in colnames:
        if str(count) in word:
            print(f'\n{count}', end=' ')
            count += 1
        else:
            print(word, end=' ')


def try_mkdir(path, skip=False, verbose=True):
    """Try to make given directory

    parameters
    ----------
    path: str
    skip : bool (optional)
        do nothing if directory already exists
        if skip=false, will ask to overwrite an existing directory
    verbose : bool (optional)
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


def ensure_temp_dir_exists(model, runs_path=None, runs_prefix='run_', verbose=True):
    """Ensure temp directory exists (create if not)
    """
    temp_path = paths.temp_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    try_mkdir(temp_path, skip=True, verbose=verbose)


def fast_line_count(filepath):
    """Efficiently find the number of lines in a file
    """
    lines = 0
    buf_size = 1024 * 1024

    with open(filepath, 'rb') as f:
        read_f = f.raw.read
        buf = read_f(buf_size)

        while buf:
            lines += buf.count(b'\n')
            buf = read_f(buf_size)

    return lines
