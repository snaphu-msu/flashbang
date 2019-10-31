import os
import numpy as np

# flashbang
from . import load_save

# TODO:
#   - extract mass grid from stir traj's
#   2. build snec traj
#       - For each mass, join profiles in time
#       - subtract t=0 offset
#       - get every 0.01 s
#   3. patch stir and snec
#   4. save files


def load_stir_traj(tracer_i, basename='stir2_oct8_s12.0_alpha1.25',
                   prefix='_tracer', extension='.dat',
                   path='/Users/zac/projects/codes/traj_code/data/traj_s12.0_1024'):
    """Load STIR trajectory from file
        Returns: 2D np.array
    """
    filepath = stir_traj_filepath(tracer_i, basename=basename, prefix=prefix,
                                  extension=extension, path=path)
    return np.loadtxt(filepath, skiprows=2)


def load_snec_profile(var, path='/Users/zac/projects/data/snec/mass13/Data'):
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
                       path='/Users/zac/projects/codes/traj_code/data/traj_s12.0_1024'):
    """Returns formatted filepath to stir trajectory file
    """
    filename = f'{basename}{prefix}{tracer_i}{extension}'
    return os.path.join(path, filename)


def subset_snec_profile(profile, t_end, dt):
    """Returns subset of profile for given timestep
    """
    t_idxs = subset_idxs(t_end, dt=dt)
    return profile[t_idxs, :]


def subset_idxs(t_end, dt):
    """Returns indices for subset time grid
    """
    full_time_grid = load_snec_profile('time_grid')
    n_dt = int(t_end / dt)
    time_grid = np.linspace(0, t_end, n_dt+1)
    return np.searchsorted(full_time_grid, time_grid)
