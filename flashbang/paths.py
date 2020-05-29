"""Functions that return standardised strings for paths and filenames for
    FLASH models and their extracted datafiles

For this module to work, you must set the bash environment variables:
    - FLASHBANG (path to flashbang repo)
    - FLASH_MODELS (path to directory containing FLASH models

Function naming convention:
  - "_filename" name of file only
  - "_path" full path to a directory
  - "_filepath" full path to a file (i.e., path + filename)

Model directory structure expected:
    <FLASH_MODELS>
        /<model_set> (if provided)
            /<model>
                /output

    where:
        - '.dat' and '.log' files are located in directory '<model>'
        - 'chk' and 'plt' files are located in directory 'output'

    Note:
        - name of 'output' directory can be manually set with arg 'output_dir'
"""

import os


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
def model_path(model, model_set=''):
    """Return path to model directory

    parameters
    ----------
    model : str
    model_set : str
    """
    try:
        flash_models_path = os.environ['FLASH_MODELS']
    except KeyError:
        raise EnvironmentError('Environment variable FLASH_MODELS not set. '
                               'Set path to directory containing flash models, e.g., '
                               "'export FLASH_MODELS=${HOME}/BANG/runs'")

    return os.path.join(flash_models_path, model_set, model)


def output_path(model, model_set='', output_dir='output'):
    """Return path to model output directory

    Parameters
    ----------
    model : str
    model_set : str
    output_dir : str
    """
    m_path = model_path(model, model_set=model_set)
    return os.path.join(m_path, output_dir)


def model_cache_path(model, model_set=''):
    """Path to directory for keeping cached files

    Parameters
    ----------
    model : str
    model_set : str
    """
    path = cache_path()
    return os.path.join(path, model_set, model)


# ===============================================================
#                      Dat files
# ===============================================================
def dat_filename(run):
    """Return filename for .dat file

    Parameters
    ----------
    run : str
    """
    return f'{run}.dat'


def dat_filepath(model, run, model_set=''):
    """Return filepath to .dat file

    parameters
    ----------
    model : str
    run : str
    model_set : str
    """
    filename = dat_filename(run)
    m_path = model_path(model, model_set=model_set)
    return os.path.join(m_path, filename)


def dat_cache_filename(model, run):
    """Return filename for cached dat file

    Parameters
    ----------
    model : str
    run : str
    """
    return f'{model}_{run}_dat.pickle'


def dat_cache_filepath(model, run, model_set=''):
    """Return filepath to cached dat table

    Parameters
    ----------
    model : str
    run : str
    model_set : str
    """
    path = model_cache_path(model, model_set=model_set)
    filename = dat_cache_filename(model, run)
    return os.path.join(path, filename)


# ===============================================================
#                      Log files
# ===============================================================
def log_filename(run):
    """Return filename for .log file

    Parameters
    ----------
    run : str
    """
    return f'{run}.log'


def log_filepath(model, run, model_set=''):
    """Return filepath to .log file

    Parameters
    ----------
    model : str
    run : str
    model_set : str
    """
    filename = log_filename(run)
    m_path = model_path(model, model_set=model_set)
    return os.path.join(m_path, filename)


# ===============================================================
#                      Chk files
# ===============================================================
def chk_filename(chk, run):
    """Return filename for checkpoint (chk) file

    Parameters
    ----------
    chk : int
    run : str
    """
    return f'{run}_hdf5_chk_{chk:04d}'


def chk_filepath(chk, model, run, model_set='', o_path=None):
    """Return filepath to checkpoint file

    Parameters
    ----------
    chk : int
    model : str
    run : str
    model_set : str
    o_path : str
    """
    filename = chk_filename(chk=chk, run=run)

    if o_path is None:
        o_path = output_path(model, model_set=model_set)

    return os.path.join(o_path, filename)


# ===============================================================
#                      chk_table
# ===============================================================
def chk_table_filename(model, run):
    """Return filename for checkpoint data-table file

    Parameters
    ----------
    model : str
    run : str
    """
    return f'{model}_{run}_chk_table.pickle'


def chk_table_filepath(model, run, model_set=''):
    """Return filepath to checkpoint data-table file

    Parameters
    ----------
    model : str
    run : str
    model_set : str
    """
    path = model_cache_path(model, model_set=model_set)
    filename = chk_table_filename(model=model, run=run)
    return os.path.join(path, filename)


# ===============================================================
#                      Profiles
# ===============================================================
def multiprofile_filename(model, run):
    """Return filename for pre-extracted multiprofile

    Parameters
    ----------
    model : str
    run : str
    """
    return f'{model}_{run}_multiprofile.nc'


def multiprofile_filepath(model, run, model_set=''):
    """Return filepath for pre-extracted multiprofile

    Parameters
    ----------
    model : str
    run : str
    model_set : str
    """
    path = model_cache_path(model, model_set=model_set)
    filename = multiprofile_filename(model=model, run=run)
    return os.path.join(path, filename)


def profile_filename(chk, model, run):
    """Return filename for pre-extracted profile

    Parameters
    ----------
    chk : int
    model : str
    run : str
    """
    return f'{model}_{run}_profile_{chk:04d}.nc'


def profile_filepath(chk, model, run, model_set=''):
    """Return filepath to pre-extracted profile

    Parameters
    ----------
    chk : int
    model : str
    run : str
    model_set : str
    """
    path = model_cache_path(model, model_set=model_set)
    filename = profile_filename(chk, model=model, run=run)
    return os.path.join(path, filename)


# ===============================================================
#                      Timesteps
# ===============================================================
def timesteps_filename(model, run):
    """Return filename for pre-extracted timestep table

    parameters
    ----------
    model : str
    run : str
    """
    return f'{model}_{run}_timesteps.pickle'


def timesteps_filepath(model, run, model_set=''):
    """Return filename for pre-extracted timestep table

    parameters
    ----------
    model : str
    run : str
    model_set : str
    """
    path = model_cache_path(model, model_set=model_set)
    filename = timesteps_filename(model=model, run=run)
    return os.path.join(path, filename)


# ===============================================================
#                      Mass Tracers
# ===============================================================
def tracers_filename(model, run):
    """Return filename for pre-extracted mass tracers dataset

    parameters
    ----------
    model : str
    run : str
    """
    return f'{model}_{run}_tracers.nc'


def tracers_filepath(model, run, model_set=''):
    """Return filepath for pre-extracted mass tracers dataset

    parameters
    ----------
    model : str
    run : str
    model_set : str
    """
    path = model_cache_path(model, model_set=model_set)
    filename = tracers_filename(model=model, run=run)
    return os.path.join(path, filename)
