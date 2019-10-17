"""Functions that return strings for paths and files

Naming convention:
  - "_filename" name of file only
  - "_path" full path to a directory
  - "_filepath" full path to a file (i.e., path + filename)
"""

import os

bangpy_path = os.environ['BANGPY']
models_path = os.environ['BANG_MODELS']


def config_filepath(name='config'):
    """Returns path to config file

    parameters
    ----------
    name : str (optional)
        base name of config file
        defaults to 'config' (for file 'config.ini')
    """
    return os.path.join(bangpy_path, f'{name}.ini')


def model_path(model, runs_path=None, prefix='run_'):
    """Returns path to model directory

    parameters
    ----------
    model : str
        name of flash model
    runs_path : str (optional)
        path to directory containing all flash models
        defaults to environment variable BANG_MODELS
    prefix : str (optional)
         prefix of model directory
         defaults to 'run_'
    """
    if runs_path is None:
        runs_path = models_path

    return os.path.join(runs_path, f'{prefix}{model}')
