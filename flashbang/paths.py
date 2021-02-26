"""Functions that return standardised strings for paths and filenames for
    FLASH models and their extracted datafiles

For this module to work, you must set the bash environment variables:
    - FLASHBANG (path to flashbang repo)
    - FLASH_MODELS (path to directory containing FLASH models)

Function naming convention:
  - "_filename" name of file only
  - "_path" full path to a directory
  - "_filepath" full path to a file (i.e., path + filename)

Expected model directory structure
----------------------------------
$FLASH_MODELS
│
└───<model_set>
|   |
|   └───<model>
|   │   │   <run>.dat
|   │   │   <run>.log
|   │   │   ...
|   │   │
|   │   └───output
|   │       │   <run>_hdf5_chk_0000
|   │       │   <run>_hdf5_chk_0001
|   │       │   ...
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


def output_path(model, model_set):
    """Return path to model output directory

    Parameters
    ----------
    model : str
    model_set : str
    """
    path = model_path(model, model_set=model_set)
    return os.path.join(path, 'output')


# ===============================================================
#                      FLASH files
# ===============================================================
def flash_filename(name, run, chk=None):
    """Return filename for raw FLASH data file

    Parameters
    ----------
    name : str
    chk : int
    run : str
    """
    chk = check_chk(chk=chk, name=name, requires_chk=['chk'])

    filenames = {
        'dat': f'{run}.dat',
        'log': f'{run}.log',
        'chk': f'{run}_hdf5_chk_{chk:04d}',
    }

    if name not in filenames:
        raise ValueError(f"'{name}' not a valid FLASH file type")

    return filenames[name]


def flash_filepath(name, run, model, model_set, chk=None):
    """Return filepath to raw FLASH data file

    Parameters
    ----------
    name : str
    chk : int
    run : str
    model : str
    model_set : str
    """
    if name == 'chk':
        path = output_path(model, model_set=model_set)
    else:
        path = model_path(model, model_set=model_set)

    filename = flash_filename(name, run=run, chk=chk)
    return os.path.join(path, filename)


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


def cache_filename(name, run, model, chk=None):
    """Return filename for cache file

    Parameters
    ----------
    name : str
    run : str
    model : str
    chk : int
    """
    chk = check_chk(chk=chk, name=name, requires_chk=['profile'])

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


def cache_filepath(name, run, model, model_set, chk=None):
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


def check_chk(chk, name, requires_chk):
    """Check that chk is provided when required

    Parameters
    ----------
    chk : int
    name : str
    requires_chk : [str]
    """
    if chk is None:
        if name in requires_chk:
            raise ValueError(f"must provide chk for '{name}'")
        else:
            chk = -1  # dummy value required for string substitution

    return chk
