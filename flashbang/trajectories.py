import os
import sys
import numpy as np
from astropy import units

# TODO:
#   4. save files

# Global variables:
g2msun = units.g.to(units.Msun)


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

