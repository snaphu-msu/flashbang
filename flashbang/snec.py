"""Functions for extracting SNEC model data
"""
import numpy as np

# flashbang
from .strings import printv


def reduce_snec_profile(profile_dict):
    """Reduce given profile dictionary into a 2D nparray
        Returns: profile_array, timesteps, mass_grid

    parameters
    ----------
    profile_dict : {}
        Dictionary containing profile data, as returned from load_snec_xg()
    """
    timesteps = np.array(list(profile_dict.keys()))
    n_time = len(timesteps)

    mass_grid = profile_dict[timesteps[0]][:, 0]
    n_mass = len(mass_grid)

    profile_array = np.zeros((n_time, n_mass))

    for i, key in enumerate(timesteps):
        profile_array[i, :] = profile_dict[key][:, 1]

    return profile_array, timesteps, mass_grid


def load_snec_xg(filepath, verbose=True):
    """Load mass tracers from SNEC output .xg file, returns as dict

    parameters
    ----------
    filepath : str
    verbose : bool
    """
    printv(f'Loading: {filepath}', verbose)
    n_lines = fast_line_count(filepath)

    profile = {}
    with open(filepath, 'r') as rf:
        count = 0
        for line in rf:
            printv(f'\r{100 * count / n_lines:.1f}%', verbose, end='')
            cols = line.split()

            # Beginning of time data - make key for this time
            if 'Time' in line:
                timesteps = float(cols[-1])
                profile[timesteps] = []

            # In time data -- build x,y arrays
            elif len(cols) == 2:
                profile[timesteps].append(np.fromstring(line, sep=' '))

            # End of time data (blank line) -- make list into array
            else:
                profile[timesteps] = np.array(profile[timesteps])
            count += 1

    printv('\n', verbose)
    return profile


def fast_line_count(filepath):
    """Efficiently find the number of lines in a file

    parameters
    ----------
    filepath: str
    """
    lines = 0
    buf_size = 1024 * 1024

    with open(filepath, 'rb') as f:
        read_f = f.raw.read
        buf = read_f(buf_size)

        while buf:
            lines += buf.count(b'\n')
            buf = read_f(buf_size)

    return lines
