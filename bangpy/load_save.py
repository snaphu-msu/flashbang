"""Functions for loading/saving files
"""
import os
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
