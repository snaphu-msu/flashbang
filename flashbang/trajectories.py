import os
import numpy as np

# flashbang
from . import load_save

# TODO:
#   - extract mass grid from stir traj's
#   2. build snec traj
#       - For each mass, join profiles in time
#       - subtract t=0 offset
#   3. patch stir and snec
#   4. save files


def load_stir_traj(tracer_i, basename='stir2_oct8_s12.0_alpha1.25',
                   prefix='_tracer', extension='.dat',
                   path='/Users/zac/projects/codes/traj_code/traj_s12.0_1024'):
    """Load STIR trajectory from file
        Returns: 2D np.array
    """
    filepath = stir_traj_filepath(tracer_i, basename=basename, prefix=prefix,
                                  extension=extension, path=path)
    return np.loadtxt(filepath, skiprows=2)


def load_snec_profile(var, path='/Users/zac/projects/codes/traj_code/snec'):
    """Load precomputed SNEC profiles from file
    """
    filename = f'{var}.npy'
    filepath = os.path.join(path, filename)
    return np.load(filepath)


def extract_stir_mass_grid(n_traj):
    """Obtain mass grid from stir trajectory file headers
    """
    pass


def stir_traj_filepath(tracer_i, basename='stir2_oct8_s12.0_alpha1.25',
                       prefix='_tracer', extension='.dat',
                       path='/Users/zac/projects/codes/traj_code/traj_s12.0_1024'):
    """Returns formatted filepath to stir trajectory file
    """
    filename = f'{basename}{prefix}{tracer_i}{extension}'
    return os.path.join(path, filename)

