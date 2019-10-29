import os
import numpy as np

# flashbang
from . import load_save

# TODO:
#   - load snec profile array
#   - extract mass grid from stir traj's
#   2. build snec traj
#       - For each mass, join profiles in time
#   3. patch stir and snec
#   4. save files


def load_stir_traj(tracer_i, basename='stir2_oct8_s12.0_alpha1.25',
                   prefix='_tracer', extension='.dat',
                   path='/Users/zac/projects/codes/traj_code/traj_s12.0_1024'):
    """Load STIR trajectory from file
        Returns: 2D np.array
    """
    filename = f'{basename}{prefix}{tracer_i}{extension}'
    filepath = os.path.join(path, filename)
    return np.loadtxt(filepath, skiprows=2)


def load_snec_profile(var, path='/Users/zac/projects/codes/traj_code/snec'):
    """Load precomputed SNEC profiles from file
    """
    filename = f'{var}.npy'
    filepath = os.path.join(path, filename)
    return np.load(filepath)


def load_snec_time_grid():
    """Load time grid from file
    """
    pass


def load_snec_mass_grid():
    """Load snec mass grid from file
    """
    pass
