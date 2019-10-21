"""Functions for loading/saving files
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


def load_config(filepath, verbose=True):
    """Load .ini config file and return as dict

    parameters
    ----------
    filepath: str
    verbose : bool
    """
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


def load_dat(filepath, cols_dict, verbose=True):
    """Loads .dat file and returns as dict of quantities

    parameters
    ----------
    filepath: str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    verbose : bool
    """
    printv(f'Loading dat file: {filepath}', verbose=verbose)
    dat = {}

    for key, idx_1 in cols_dict.items():
        index = idx_1 - 1
        dat[key] = np.loadtxt(filepath, usecols=index)

    return dat


def extract_profile(basename, chk_i, model, xmax=1e12, output_dir='output',
                    runs_path=None, runs_prefix='run_', o_path=None,
                    params=('temp', 'dens', 'pres'), reload=False,
                    save=True, verbose=True):
    """Extracts and returns profile dict from checkpoint file

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
    def reload_chk():
        return load_chk(basename, model=model, chk_i=chk_i,
                        output_dir=output_dir, runs_path=runs_path,
                        runs_prefix=runs_prefix, o_path=o_path)
    profile = {}
    loaded_successfully = False
    if not reload:
        try:
            profile = load_profile(basename, chk_i=chk_i, model=model,
                                   runs_path=runs_path, runs_prefix=runs_prefix,
                                   verbose=verbose)
            loaded_successfully = True
        except FileNotFoundError:
            pass

    if len(profile.keys()) == 0:
        chk = reload_chk()
        ray = chk.ray([0, 0, 0], [xmax, 0, 0])
        profile['x'] = ray['t'] * xmax

        for v in params:
            profile[v] = ray[v]

    if save and not loaded_successfully:
        save_profile(profile, basename=basename, chk_i=chk_i, model=model,
                     runs_path=runs_path, runs_prefix=runs_prefix, verbose=verbose)
    return profile


def save_profile(profile, basename, chk_i, model, runs_path=None,
                 runs_prefix='run_', verbose=True):
    """Saves profile to file for faster loading

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
    filepath = paths.profile_filepath(basename, model=model, chk_i=chk_i,
                                      runs_path=runs_path, runs_prefix=runs_prefix)
    printv(f'Saving: {filepath}', verbose)
    pickle.dump(profile, open(filepath, 'wb'))


def load_profile(basename, chk_i, model, runs_path=None,
                 runs_prefix='run_', verbose=True):
    """Saves profile to file for faster loading

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
    """Loads checkpoint file for given model

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
    return yt.load(filepath)


def find_chk(path, match_str='hdf5_chk_', n_digits=4):
    """Returns list of checkpoint (chk) files available in given directory
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
