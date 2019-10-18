"""Functions that return strings for paths and files

Naming convention:
  - "_filename" name of file only
  - "_path" full path to a directory
  - "_filepath" full path to a file (i.e., path + filename)

Expected directory structure:
[TODO]
"""

import os

# Bash environment variables, need to set these first
bangpy_path = os.environ['BANGPY']
models_path = os.environ['BANG_MODELS']


def dat_filename(job_name):
    """Returns filename for .dat file
    """
    return f'{job_name}.dat'


def dat_filepath(job_name, model, runs_path=None, runs_prefix='run_'):
    """Returns filepath to .dat file

    parameters
    ----------
    job_name : str
    model : str
    runs_path : str
        see model_path()
    runs_prefix : str
        see model_path()
    """
    filename = dat_filename(job_name)
    path = model_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    return os.path.join(path, filename)


def chk_filename(job_name, chk_i):
    """Returns filename for checkpoint (chk) file
    """
    return f'{job_name}_hdf5_chk_{chk_i:04d}'


def chk_filepath(job_name, model, chk_i, output_dir='output',
                 runs_path=None, runs_prefix='run_'):
    """Returns filepath to checkpoint file
    """
    filename = chk_filename(job_name, chk_i=chk_i)
    path = model_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    return os.path.join(path, output_dir, filename)


def config_filepath(name='default'):
    """Returns path to config file

    parameters
    ----------
    name : str (optional)
        base name of config file
        defaults to 'default' (for file 'default.ini')
    """
    return os.path.join(bangpy_path, 'bangpy', 'config', f'{name}.ini')


def model_path(model, runs_path=None, runs_prefix='run_'):
    """Returns path to model directory

    parameters
    ----------
    model : str
        name of flash model
    runs_path : str (optional)
        path to directory containing all flash models
        defaults to environment variable BANG_MODELS
    runs_prefix : str (optional)
         prefix of model directory
         defaults to 'run_'
    """
    if runs_path is None:
        runs_path = models_path

    return os.path.join(runs_path, f'{runs_prefix}{model}')
