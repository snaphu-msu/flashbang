import os
import numpy as np
import matplotlib.pyplot as plt
import yt
import configparser
import ast

# bangpy
from . import paths

# TODO:
#   - extract and save subsets of profiles (for faster loading)
#   - load .dat file
#   - plotly slider
#   -


class BangSim:
    def __init__(self, model, runs_path=None, xmax=1e12, config='config',
                 dim=1, job_name=None, output_dir='output', verbose=True):
        self.verbose = verbose

        if runs_path is None:
            runs_path = paths.model_path(model=model, runs_path=runs_path)
        self.path = os.path.join(runs_path, f'run_{model}')
        self.output_path = os.path.join(self.path, output_dir)

        self.model = model
        self.job_name = job_name
        self.dim = dim
        self.xmax = xmax

        self.config = None
        self.dat = None
        self.chk = None
        self.ray = None
        self.x = None

    def printv(self, string):
        if self.verbose:
            print(string)

    def load_config(self, config='config'):
        """Load config file
        """
        config_filepath = paths.config_filepath(name=config)
        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f'Config file not found: {config_filepath}')

        self.printv(f'Loading config: {config_filepath}')
        ini = configparser.ConfigParser()
        ini.read(config_filepath)

        config = {}
        for section in ini.sections():
            config[section] = {}
            for option in ini.options(section):
                config[section][option] = ast.literal_eval(ini.get(section, option))

        self.config = config

    def load_dat(self):
        """Load .dat file
        """
        f_name = f'{self.job_name}.dat'
        f_path = os.path.join(self.path, f_name)
        self.dat = np.loadtxt(f_path, usecols=[0])

    def load_chk(self, step):
        """Load checkpoint data file
        """
        f_name = f'{self.job_name}_hdf5_chk_{step:04d}'
        f_path = os.path.join(self.output_path, f_name)
        self.chk = yt.load(f_path)
        self.ray = self.chk.ray([0, 0, 0], [self.xmax, 0, 0])
        self.x = self.ray['t'] * self.xmax

    def plot(self, var, y_log=True, x_log=True):
        """Plot given variable
        """
        fig, ax = plt.subplots()
        ax.plot(self.x, self.ray[var])

        if y_log:
            ax.set_yscale('log')
        if x_log:
            ax.set_xscale('log')

