"""Functions that return standardised strings for paths and files for
    FLASH models

For this module to work, you must set the bash environment variables:
    - FLASHBANG (path to flashbang repo)
    - FLASH_MODELS (path to directory containing FLASH models

Function naming convention:
  - "_filename" name of file only
  - "_path" full path to a directory
  - "_filepath" full path to a file (i.e., path + filename)

Default model directory structure expected:
    <FLASH_MODELS>
        /<model>
            /output

    where:
        - '.dat' and '.log' files are located in directory '<model>'
        - 'chk' and 'plt' files are located in directory 'output'

    Note:
        - name of 'output' directory can be manually set with arg 'output_dir'
"""

import os

# TODO:
#   - deprecate output_dir, o_path (replace with environment variables)


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
    try:
        flashbang_path = os.environ['FLASHBANG']
    except KeyError:
        raise EnvironmentError('Environment variable FLASHBANG not set. '
                               'Set path to flashbang directory, e.g., '
                               "'export FLASHBANG=${HOME}/codes/flashbang'")

    return os.path.join(flashbang_path, 'flashbang', 'config', f'{name}.ini')


# ===============================================================
#                      Models
# ===============================================================
def model_path(model):
    """Return path to model directory

    parameters
    ----------
    model : str
        name of flash model
    """
    try:
        flash_models_path = os.environ['FLASH_MODELS']
    except KeyError:
        raise EnvironmentError('Environment variable FLASH_MODELS not set. '
                               'Set path to directory containing flash models, e.g., '
                               "'export FLASH_MODELS=${HOME}/BANG/runs'")

    return os.path.join(flash_models_path, model)


def temp_path(model):
    """Path to directory for temporary file saving
    """
    m_path = model_path(model)
    return os.path.join(m_path, 'temp')


def output_path(model, output_dir='output'):
    """Return path to model output directory
    """
    m_path = model_path(model)
    return os.path.join(m_path, output_dir)


# ===============================================================
#                      Dat files
# ===============================================================
def dat_filename(run):
    """Return filename for .dat file
    """
    return f'{run}.dat'


def dat_filepath(model, run='run'):
    """Return filepath to .dat file

    parameters
    ----------
    run : str
    model : str
    """
    filename = dat_filename(run)
    m_path = model_path(model)
    return os.path.join(m_path, filename)


def dat_temp_filename(run):
    """Return filename for temporary (cached) dat file
    """
    return f'{run}_dat.pickle'


def dat_temp_filepath(model, run='run'):
    """Return filepath to reduced dat table
    """
    path = temp_path(model)
    filename = dat_temp_filename(run)
    return os.path.join(path, filename)


# ===============================================================
#                      Log files
# ===============================================================
def log_filename(run):
    """Return filename for .log file
    """
    return f'{run}.log'


def log_filepath(model, run='run'):
    """Return filepath to .log file
    """
    filename = log_filename(run)
    m_path = model_path(model)
    return os.path.join(m_path, filename)


# ===============================================================
#                      Chk files
# ===============================================================
def chk_filename(chk, run):
    """Return filename for checkpoint (chk) file
    """
    return f'{run}_hdf5_chk_{chk:04d}'


def chk_filepath(chk, model, run='run', output_dir='output', o_path=None):
    """Return filepath to checkpoint file
    """
    filename = chk_filename(chk=chk, run=run)
    if o_path is None:
        o_path = output_path(model, output_dir=output_dir)
    return os.path.join(o_path, filename)


# ===============================================================
#                      Profiles
# ===============================================================
def profile_filename(chk, run):
    """Return filename for pre-extracted profile
    """
    return f'{run}_profile_{chk:04d}.pickle'


def profile_filepath(chk, model, run='run'):
    """Return filepath to pre-extracted profile
    """
    path = temp_path(model)
    filename = profile_filename(chk, run)
    return os.path.join(path, filename)


# ===============================================================
#                      Timesteps
# ===============================================================
def timesteps_filename(model, run='run'):
    """Return filename for pre-extracted timestep table

    parameters
    ----------
    model : str
    run : str
    """
    return f'{run}_{model}_timesteps.pickle'


def timesteps_filepath(model, run='run'):
    """Return filename for pre-extracted timestep table

    parameters
    ----------
    model : str
    run : str
    """
    path = temp_path(model)
    filename = timesteps_filename(model=model, run=run)
    return os.path.join(path, filename)
