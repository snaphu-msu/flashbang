"""Functions for loading/saving files
"""
import os
import numpy as np
import configparser
import ast
import subprocess
import sys

# bangpy
from .strings import printv


def try_mkdir(path, skip=False, verbose=True):
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
