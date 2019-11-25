import os
import sys
import numpy as np
from scipy.interpolate import interp1d
from astropy import units

# flashbang
from .strings import printv

# TODO:
#   4. save files

# Global variables:
g2msun = units.g.to(units.Msun)


def pipeline(n_tracers=100, n_skip=1, run='concat_stir_snec',
             path='/Users/zac/projects/codes/traj_code/data/concat',
             snec_tracers=None):
    """Join and save tracers
    """
    if snec_tracers is None:
        snec_tracers = build_snec_tracers()

    for i in range(n_tracers):
        sys.stdout.write(f'\rJoin tracer {i+1}/{n_tracers}')
        tracer = join_tracers(snec_tracers, mass_i=i, n_skip=n_skip)

        filename = f'{run}_tracer{i}.dat'
        filepath = os.path.join(path, filename)
        np.savetxt(filepath, tracer, fmt='%.10e', delimiter='    ')

    sys.stdout.write('\n')


def join_tracers(snec_tracers, mass_i, n_skip=1):
    """Join stir and snec tracer

    snec_tracers
        as output by build_snec_tracers
    """
    stir_tracer = load_stir_traj(mass_i)
    snec_tracer = np.array(snec_tracers[mass_i, n_skip:, :])

    t_offset = stir_tracer[-1, 0]
    snec_tracer[:, 0] += t_offset  # shift to stir time

    return np.concatenate([stir_tracer, snec_tracer])


def build_snec_tracers(t_end=20, dt=0.01, n_traj=100,
                       var_list=('temp', 'rho', 'radius', 'ye'), verbose=True):
    """Build snec mass tracers to append to stir tracers
    """
    # Get reduced snec time grid
    full_time_grid = load_snec_profile('time_grid')
    sub_idxs = subset_idxs(t_end=t_end, dt=dt)
    time_grid = full_time_grid[sub_idxs]

    mass_grid = extract_stir_mass_grid(n_traj=n_traj)

    # map snec profiles onto stir mass grid
    map_profiles = dict.fromkeys(var_list)
    for var in var_list:
        map_profiles[var] = map_snec_grid(var, mass_grid=mass_grid)

    n_time = len(time_grid)
    n_mass = len(mass_grid)
    n_vars = len(var_list) + 1  # one extra for time

    tracers = np.full([n_mass, n_time, n_vars], np.nan)

    printv('Building mass tracers from mapped profiles', verbose)
    for i in range(n_mass):
        tracers[i, :, 0] = time_grid

        for j, var in enumerate(var_list):
            tracers[i, :, j+1] = map_profiles[var][:, i]

    return tracers


def load_all_stir_tracers(n_tracers, run='stir2_oct8_s12.0_alpha1.25',
                          prefix='_tracer', extension='.dat',
                          path='/Users/zac/projects/codes/traj_code/data/traj_s12.0_1024'):
    """Load all stir tracers and return as single array
    """
    t0 = load_stir_traj(0, run=run, prefix=prefix, extension=extension,
                        path=path)
    n_time, n_var = t0.shape
    tracers = np.full([n_tracers, n_time, n_var], np.nan)

    for i in range(n_tracers):
        sys.stdout.write(f'\rloading stir tracer: {i+1}/{n_tracers}')
        tracer = load_stir_traj(i, run=run, prefix=prefix, extension=extension,
                                path=path)
        tracers[i, :, :] = tracer

    sys.stdout.write('\n')
    return tracers


def load_stir_traj(tracer_i, run='stir2_oct8_s12.0_alpha1.25',
                   prefix='_tracer', extension='.dat', skiprows=2,
                   path='/Users/zac/projects/codes/traj_code/data/traj_s12.0_1024'):
    """Load STIR trajectory from file
        Returns: 2D np.array
    """
    filepath = stir_traj_filepath(tracer_i, run=run, prefix=prefix,
                                  extension=extension, path=path)
    return np.loadtxt(filepath, skiprows=skiprows)


def load_snec_profile(var, path='/Users/zac/projects/data/snec/mass13/Data'):
    """Load precomputed SNEC profiles from file
    """
    filename = f'{var}.npy'
    filepath = os.path.join(path, filename)
    return np.load(filepath)


def map_snec_grid(var, mass_grid):
    """Interpolate snec profile onto stir tracer mass grid

    parameters
    ----------
    var : str
        profile variable to map (e.g., radius)
    mass_grid : []
        1D mass grid to map onto
    """
    print(f'Mapping var={var} profile from snec onto stir mass grid')
    snec_mass_grid = g2msun * load_snec_profile('mass_grid')
    snec_profile = load_snec_profile(f'sub_{var}')

    n_time = len(snec_profile[:, 0])
    n_mass = len(mass_grid)
    mapped = np.full((n_time, n_mass), np.nan)

    for i in range(n_time):
        snec_func = interp1d(snec_mass_grid, snec_profile[i, :])
        mapped[i, :] = snec_func(mass_grid)

    return mapped


def extract_stir_mass_grid(n_traj=100):
    """Obtain mass grid from stir trajectory file headers
    """
    mass_grid = []
    for i in range(n_traj):
        filepath = stir_traj_filepath(tracer_i=i)
        with open(filepath, 'r') as f:
            line = f.readline()
            mass_grid += [float(line.split()[3])]

    return np.array(mass_grid)


def stir_traj_filepath(tracer_i, run='stir2_oct8_s12.0_alpha1.25',
                       prefix='_tracer', extension='.dat',
                       path='/Users/zac/projects/codes/traj_code/data/traj_s12.0_1024'):
    """Returns formatted filepath to stir trajectory file
    """
    filename = f'{run}{prefix}{tracer_i}{extension}'
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
