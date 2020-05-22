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
#   - replace default run=None


# ===============================================================
#                      Flashbang
# ===============================================================
def flashbang_path():
    """Return path to flashbang repo
    """
    try:
        path = os.environ['FLASHBANG']
    except KeyError:
        raise EnvironmentError('Environment variable FLASHBANG not set. '
                               'Set path to flashbang directory, e.g., '
                               "'export FLASHBANG=${HOME}/codes/flashbang'")
    return path


def config_filepath(name=None):
    """Return path to config file

    parameters
    ----------
    name : str (optional)
        base name of config file
        defaults to 'default' (for config file 'default.ini')
    """
    path = flashbang_path()

    if name is None:
        name = 'default'

    return os.path.join(path, 'flashbang', 'config', f'{name}.ini')


def cache_path():
    """Path to directory for cached files
    """
    path = flashbang_path()
    return os.path.join(path, 'cache')


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


def output_path(model, output_dir='output'):
    """Return path to model output directory
    """
    m_path = model_path(model)
    return os.path.join(m_path, output_dir)


def model_cache_path(model):
    """Path to directory for keeping cached files
    """
    path = cache_path()
    return os.path.join(path, model)


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


def dat_cache_filename(model, run):
    """Return filename for cached dat file
    """
    return f'{model}_{run}_dat.pickle'


def dat_cache_filepath(model, run='run'):
    """Return filepath to cached dat table
    """
    path = model_cache_path(model)
    filename = dat_cache_filename(model, run)
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


def chk_filepath(chk, model, run='run', o_path=None):
    """Return filepath to checkpoint file
    """
    filename = chk_filename(chk=chk, run=run)

    if o_path is None:
        o_path = output_path(model)

    return os.path.join(o_path, filename)


# ===============================================================
#                      chk_table
# ===============================================================
def chk_table_filename(model, run):
    """Return filename for checkpoint data-table file
    """
    return f'{model}_{run}_chk_table.pickle'


def chk_table_filepath(model, run='run'):
    """Return filepath to checkpoint data-table file
    """
    path = model_cache_path(model)
    filename = chk_table_filename(model=model, run=run)
    return os.path.join(path, filename)


# ===============================================================
#                      Profiles
# ===============================================================
def multiprofile_filename(model, run):
    """Return filename for pre-extracted multiprofile
    """
    return f'{model}_{run}_multiprofile.nc'


def multiprofile_filepath(model, run='run'):
    """Return filepath for pre-extracted multiprofile
    """
    path = model_cache_path(model)
    filename = multiprofile_filename(model=model, run=run)
    return os.path.join(path, filename)


def profile_filename(chk, model, run):
    """Return filename for pre-extracted profile
    """
    return f'{model}_{run}_profile_{chk:04d}.nc'


def profile_filepath(chk, model, run='run'):
    """Return filepath to pre-extracted profile
    """
    path = model_cache_path(model)
    filename = profile_filename(chk, model=model, run=run)
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
    return f'{model}_{run}_timesteps.pickle'


def timesteps_filepath(model, run='run'):
    """Return filename for pre-extracted timestep table

    parameters
    ----------
    model : str
    run : str
    """
    path = model_cache_path(model)
    filename = timesteps_filename(model=model, run=run)
    return os.path.join(path, filename)


# ===============================================================
#                      Mass Tracers
# ===============================================================
def tracers_filename(model, run='run'):
    """Return filename for pre-extracted mass tracers dataset

    parameters
    ----------
    model : str
    run : str
    """
    return f'{model}_{run}_tracers.nc'


def tracers_filepath(model, run='run'):
    """Return filepath for pre-extracted mass tracers dataset

    parameters
    ----------
    model : str
    run : str
    """
    path = model_cache_path(model)
    filename = tracers_filename(model=model, run=run)
    return os.path.join(path, filename)
