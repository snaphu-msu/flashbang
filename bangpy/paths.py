"""Functions that return standardised strings for paths and files of
    BANG (FLASH) models, and also bangpy stuff

For this module to work, you must set the bash environment variables:
    - BANGPY (path to bangpy)
    - BANG_MODELS (path to location of Bang (Flash) models, i.e. BANG/runs)

Function naming convention:
  - "_filename" name of file only
  - "_path" full path to a directory
  - "_filepath" full path to a file (i.e., path + filename)

Expected directory structure:
    BANG/runs/run_[model]/output/

    where:
        - '.dat' and '.log' files are in directory 'run_[model]'
        - 'chk' and 'plt' files in directory 'output'

    Note:
        - path to 'runs' directory can be set with arg 'runs_path'
        - prefix to model directory, 'run_', can be set with arg 'runs_prefix'
        - name of 'output' directory can be set with arg 'output_dir'
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
    m_path = model_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    return os.path.join(m_path, filename)


def chk_filename(job_name, chk_i):
    """Returns filename for checkpoint (chk) file
    """
    return f'{job_name}_hdf5_chk_{chk_i:04d}'


def chk_filepath(job_name, model, chk_i, output_dir='output',
                 runs_path=None, runs_prefix='run_', o_path=None):
    """Returns filepath to checkpoint file
    """
    filename = chk_filename(job_name, chk_i=chk_i)
    if o_path is None:
        o_path = output_path(model, output_dir=output_dir, runs_path=runs_path,
                             runs_prefix=runs_prefix)
    return os.path.join(o_path, filename)


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


def output_path(model, output_dir='output', runs_path=None,
                runs_prefix='run_'):
    """Returns path to model output directory
    """
    m_path = model_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    return os.path.join(m_path, output_dir)
