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
import configparser
import ast
import subprocess
import sys
import yt
import pickle

# bangpy
from . import paths
from .strings import printv

#  TODO:
#   - pickle .dat file


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


def get_dat(basename, model, cols_dict, runs_path=None, runs_prefix='run_',
            verbose=True, save=True, reload=False):
    """Get reduced .dat data, as contained in .dat file

    Returns : dict of 1D quantities

    parameters
    ----------
    basename: str
    model : str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    runs_path : str
    runs_prefix : str
    verbose : bool
    save : bool
    reload : bool
    """
    dat = {}
    dat_exists = False

    if not reload:
        try:
            dat = load_dat(basename, model=model, runs_path=runs_path,
                           runs_prefix=runs_prefix, verbose=verbose)
            dat_exists = True
        except FileNotFoundError:
            pass

    if len(dat.keys()) == 0:
        dat = extract_dat(basename, model=model, runs_path=runs_path, runs_prefix=runs_prefix,
                          cols_dict=cols_dict)

    if save and not dat_exists:
        save_dat(dat, basename=basename, model=model, runs_path=runs_path,
                 runs_prefix=runs_prefix, verbose=verbose)
    return dat


def extract_dat(basename, model, cols_dict, runs_path=None, runs_prefix='run_',
                verbose=True):
    """Extract and reduce data from .dat file

    Returns : dict of 1D quantities

    parameters
    ----------
    basename: str
    model : str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    runs_path : str
    runs_prefix : str
    verbose : bool
    """
    # TODO: allow pickle saving/loading
    filepath = paths.dat_filepath(basename=basename, model=model, runs_path=runs_path,
                                  runs_prefix=runs_prefix)

    printv(f'Loading dat file: {filepath}', verbose=verbose)
    idxs = []
    keys = []

    for key, idx_1 in cols_dict.items():
        idxs += [idx_1 - 1]
        keys += [key]

    dat_raw = np.loadtxt(filepath, usecols=idxs)
    dat = {}
    for i, key in enumerate(keys):
        dat[key] = dat_raw[:, i]

    return dat


def save_dat(dat, basename, model, runs_path=None,
             runs_prefix='run_', verbose=True):
    """Save extracted .dat properties to file, for faster loading

    parameters
    ----------
    dat : {}
        dictionary of arrays, as extracted by load_dat
    basename : str
    model : str
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    """
    ensure_temp_dir_exists(model, runs_path=runs_path, runs_prefix=runs_prefix,
                           verbose=verbose)
    filepath = paths.dat_temp_filepath(basename, model=model, runs_path=runs_path,
                                       runs_prefix=runs_prefix)

    printv(f'Saving: {filepath}', verbose)
    pickle.dump(dat, open(filepath, 'wb'))


def load_dat(basename, model, runs_path=None,
             runs_prefix='run_', verbose=True):
    """Load profile from pre-extracted file (see: save_profile)

    parameters
    ----------
    basename : str
    model : str
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    """
    filepath = paths.dat_temp_filepath(basename, model=model, runs_path=runs_path,
                                       runs_prefix=runs_prefix)
    printv(f'Loading: {filepath}', verbose)
    return pickle.load(open(filepath, 'rb'))


def get_profile(basename, chk_i, model, xmax=1e12, output_dir='output',
                runs_path=None, runs_prefix='run_', o_path=None,
                params=('temp', 'dens', 'pres'), reload=False,
                save=True, verbose=True):
    """Get reduced radial profile, as contained in checkpoint file

    Returns : dictionary of 1D arrays

    parameters
    ----------
    basename : str
    chk_i : int
    model : str
    xmax : float (optional)
        Return profile between radius=0 to xmax
    output_dir : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    o_path : str (optional)
    params : [] (optional)
        profile parameters to extract and return from chk file
    reload : bool (optional)
        force reload from chk file, else try to load pickled profile
    save : bool
        pickle profile to file for faster loading next time
    verbose : bool (optional)
    """
    profile = {}
    profile_exists = False

    if not reload:
        try:
            profile = load_profile(basename, chk_i=chk_i, model=model, runs_path=runs_path,
                                   runs_prefix=runs_prefix, verbose=verbose)
            profile_exists = True
        except FileNotFoundError:
            pass

    if len(profile.keys()) == 0:
        profile = extract_profile(basename, chk_i=chk_i, model=model, xmax=xmax,
                                  output_dir=output_dir, runs_path=runs_path,
                                  runs_prefix=runs_prefix, o_path=o_path, params=params)

    if save and not profile_exists:
        save_profile(profile, basename=basename, chk_i=chk_i, model=model,
                     runs_path=runs_path, runs_prefix=runs_prefix, verbose=verbose)
    return profile


def extract_profile(basename, chk_i, model, xmax=1e12, output_dir='output',
                    runs_path=None, runs_prefix='run_', o_path=None,
                    params=('temp', 'dens', 'pres')):
    """Extract and reduce profile data from chk file

    Returns : dictionary of 1D arrays

    parameters
    ----------
    basename : str
    chk_i : int
    model : str
    xmax : float (optional)
        Return profile between radius=0 to xmax
    output_dir : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    o_path : str (optional)
    params : [] (optional)
        profile parameters to extract and return from chk file
    """
    profile = {}
    chk = load_chk(basename, model=model, chk_i=chk_i, output_dir=output_dir,
                   runs_path=runs_path, runs_prefix=runs_prefix, o_path=o_path)
    ray = chk.ray([0, 0, 0], [xmax, 0, 0])
    profile['x'] = ray['t'] * xmax

    for v in params:
        profile[v.strip()] = np.array(ray[v])

    return profile


def save_profile(profile, basename, chk_i, model, runs_path=None,
                 runs_prefix='run_', verbose=True):
    """Save profile to file for faster loading

    parameters
    ----------
    profile : dict
    basename : str
    chk_i : int
    model : str
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    """
    ensure_temp_dir_exists(model, runs_path=runs_path, runs_prefix=runs_prefix,
                           verbose=verbose)
    filepath = paths.profile_filepath(basename, model=model, chk_i=chk_i,
                                      runs_path=runs_path, runs_prefix=runs_prefix)

    printv(f'Saving: {filepath}', verbose)
    pickle.dump(profile, open(filepath, 'wb'))


def load_profile(basename, chk_i, model, runs_path=None,
                 runs_prefix='run_', verbose=True):
    """Load profile from pre-extracted file (see: save_profile)

    parameters
    ----------
    basename : str
    chk_i : int
    model : str
    runs_path : str (optional)
    runs_prefix : str (optional)
    verbose : bool (optional)
    """
    filepath = paths.profile_filepath(basename, model=model, chk_i=chk_i,
                                      runs_path=runs_path, runs_prefix=runs_prefix)
    printv(f'Loading: {filepath}', verbose)
    return pickle.load(open(filepath, 'rb'))


def load_chk(basename, chk_i, model, output_dir='output',
             runs_path=None, runs_prefix='run_', o_path=None):
    """Load checkpoint file for given model

    parameters
    ----------
    basename : str
    chk_i : int
    model : str
    output_dir : str (optional)
    runs_path : str (optional)
    runs_prefix : str (optional)
    o_path : str (optional)
    """
    filepath = paths.chk_filepath(basename, model=model, chk_i=chk_i,
                                  output_dir=output_dir, runs_path=runs_path,
                                  runs_prefix=runs_prefix, o_path=o_path)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'checkpoint {chk_i:04d} file does not exist: {filepath}')

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
