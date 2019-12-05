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
flashbang_path = os.environ['FLASHBANG']
models_path = os.environ['BANG_MODELS']


# ===============================================================
#                      Flashbang
# ===============================================================
def config_filepath(name='default'):
    """Return path to config file

    parameters
    ----------
    name : str (optional)
        base name of config file
        defaults to 'default' (for file 'default.ini')
    """
    return os.path.join(flashbang_path, 'flashbang', 'config', f'{name}.ini')


# ===============================================================
#                      Models
# ===============================================================
def model_path(model, runs_path=None, runs_prefix='run_'):
    """Return path to model directory

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


def temp_path(model, runs_path=None, runs_prefix='run_'):
    """Path to directory for temporary file saving
    """
    m_path = model_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    return os.path.join(m_path, 'temp')


def output_path(model, output_dir='output', runs_path=None,
                runs_prefix='run_'):
    """Return path to model output directory
    """
    m_path = model_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    return os.path.join(m_path, output_dir)


# ===============================================================
#                      Dat files
# ===============================================================
def dat_filename(run):
    """Return filename for .dat file
    """
    return f'{run}.dat'


def dat_filepath(model, run='run', runs_path=None, runs_prefix='run_'):
    """Return filepath to .dat file

    parameters
    ----------
    run : str
    model : str
    runs_path : str
        see model_path()
    runs_prefix : str
        see model_path()
    """
    filename = dat_filename(run)
    m_path = model_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    return os.path.join(m_path, filename)


def dat_temp_filename(run):
    """Return filename for temporary (cached) dat file
    """
    return f'{run}_dat.feather'


def dat_temp_filepath(model, run='run', runs_path=None, runs_prefix='run_'):
    """Return filepath to reduced dat table
    """
    path = temp_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    filename = dat_temp_filename(run)
    return os.path.join(path, filename)


# ===============================================================
#                      Log files
# ===============================================================
def log_filename(run):
    """Return filename for .log file
    """
    return f'{run}.log'


def log_filepath(model, run='run', runs_path=None, runs_prefix='run_'):
    """Return filepath to .log file
    """
    filename = log_filename(run)
    m_path = model_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    return os.path.join(m_path, filename)


# ===============================================================
#                      Chk files
# ===============================================================
def chk_filename(chk, run):
    """Return filename for checkpoint (chk) file
    """
    return f'{run}_hdf5_chk_{chk:04d}'


def chk_filepath(chk, model, run='run', output_dir='output',
                 runs_path=None, runs_prefix='run_', o_path=None):
    """Return filepath to checkpoint file
    """
    filename = chk_filename(chk=chk, run=run)
    if o_path is None:
        o_path = output_path(model, output_dir=output_dir, runs_path=runs_path,
                             runs_prefix=runs_prefix)
    return os.path.join(o_path, filename)


# ===============================================================
#                      Profiles
# ===============================================================
def profile_filename(chk, run):
    """Return filename for pre-extracted profile
    """
    return f'{run}_profile_{chk:04d}.feather'


def profile_filepath(chk, model, run='run', runs_path=None, runs_prefix='run_'):
    """Return filepath to pre-extracted profile
    """
    path = temp_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    filename = profile_filename(chk, run)
    return os.path.join(path, filename)


# ===============================================================
#                      Timesteps
# ===============================================================
def timestep_filename(model, run='run'):
    """Return filename for pre-extracted timestep table

    parameters
    ----------
    model : str
    run : str
    """
    return f'{run}_{model}_timesteps.feather'


def timestep_filepath(model, run='run', runs_path=None, runs_prefix='run_'):
    """Return filename for pre-extracted timestep table

    parameters
    ----------
    model : str
    run : str
    runs_path : str
    runs_prefix : str
    """
    path = temp_path(model, runs_path=runs_path, runs_prefix=runs_prefix)
    filename = timestep_filename(model=model, run=run)
    return os.path.join(path, filename)
