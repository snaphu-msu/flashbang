"""Functions for loading/saving files
"""
import os
import numpy as np
import configparser
import ast

# bangpy
from .strings import printv


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


def load_dat(filepath, cols_dict):
    """Loads .dat file and returns as dict of quantities

    parameters
    ----------
    filepath: str
    cols_dict : {}
        dictionary with column names and indexes (Note: 1-indexed)
    """
    dat = {}
    for key, idx_1 in cols_dict.items():
        print(idx_1)
        index = idx_1 - 1
        dat[key] = np.loadtxt(filepath, usecols=index)

    return dat
