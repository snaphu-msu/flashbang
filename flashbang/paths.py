"""Functions that return standardised strings for paths and filenames for
    FLASH models and their extracted datafiles

For this module to work, you must set the bash environment variables:
    - FLASHBANG (path to flashbang repo)
    - FLASH_MODELS (path to directory containing FLASH models)

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


# ===============================================================
#                      Models
# ===============================================================
def model_path(model, model_set):
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


def output_path(model, model_set, output_dir='output'):
    """Return path to model output directory

    Parameters
    ----------
    model : str
    model_set : str
    output_dir : str
    """
    m_path = model_path(model, model_set=model_set)
    return os.path.join(m_path, output_dir)


# ===============================================================
#                      FLASH files
# ===============================================================
def flash_filename(name, run, chk=-1):
    """Return filename for raw FLASH data file

    Parameters
    ----------
    name : str
    chk : int
    run : str
    """
    if (name == 'chk') and (chk is -1):
        raise ValueError(f"must provide chk")

    filenames = {
        'dat': f'{run}.dat',
        'log': f'{run}.log',
        'chk': f'{run}_hdf5_chk_{chk:04d}',
    }

    if name not in filenames:
        raise ValueError(f"'{name}' not a valid FLASH file type")
    
    return filenames[name]


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


def chk_filepath(chk, run, model, model_set):
    """Return filepath to checkpoint file

    Parameters
    ----------
    chk : int
    run : str
    model : str
    model_set : str
    """
    path = output_path(model, model_set=model_set)
    filename = chk_filename(chk=chk, run=run)
    return os.path.join(path, filename)


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


def dat_filepath(run, model, model_set):
    """Return filepath to .dat file

    parameters
    ----------
    run : str
    model : str
    model_set : str
    """
    filename = dat_filename(run)
    m_path = model_path(model, model_set=model_set)
    return os.path.join(m_path, filename)


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


def log_filepath(run, model, model_set):
    """Return filepath to .log file

    Parameters
    ----------
    run : str
    model : str
    model_set : str
    """
    filename = log_filename(run)
    m_path = model_path(model, model_set=model_set)
    return os.path.join(m_path, filename)


# ===============================================================
#                      Cache files
# ===============================================================
def cache_path():
    """Path to directory for cached files
    """
    path = flashbang_path()
    return os.path.join(path, 'cache')


def model_cache_path(model, model_set):
    """Path to directory for keeping cached files

    Parameters
    ----------
    model : str
    model_set : str
    """
    path = cache_path()
    return os.path.join(path, model_set, model)


def cache_filename(name, run, model, chk=-1):
    """Return filename for cache file

    Parameters
    ----------
    name : str
    run : str
    model : str
    chk : int
    """
    requires_chk = ['profile']
    if (name in requires_chk) and (chk is -1):
        raise ValueError(f"must provide chk for cache name '{name}'")

    filenames = {
        'dat': f'{model}_{run}_dat.pickle',
        'chk_table': f'{model}_{run}_chk_table.pickle',
        'multiprofile': f'{model}_{run}_multiprofile.nc',
        'profile': f'{model}_{run}_profile_{chk:04d}.nc',
        'timesteps': f'{model}_{run}_timesteps.pickle',
        'tracers': f'{model}_{run}_tracers.nc',
    }

    if name not in filenames:
        raise ValueError(f"'{name}' not a valid cache name")

    return filenames[name]


def cache_filepath(name, run, model, model_set, chk=-1):
    """Return filepath to cache file

    Parameters
    ----------
    name : str
    run : str
    model : str
    model_set : str
    chk : int
    """
    path = model_cache_path(model=model, model_set=model_set)
    filename = cache_filename(name=name, run=run, model=model, chk=chk)
    return os.path.join(path, filename)
