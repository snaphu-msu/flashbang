import os
import numpy as np
import matplotlib.pyplot as plt
import yt


class BangSim:
    def __init__(self, name, bang_path='~/projects/codes/BANG/runs', xmax=1e12,
                 dim=1, chk_name=None, output_dir='output'):
        self.path = os.path.join(bang_path, f'run_{name}')
        self.output_path = os.path.join(self.path, output_dir)
        self.name = name

        self.chk_name = chk_name
        if self.chk_name is None:
            self.chk_name = name

        self.dim = dim
        self.xmax = xmax
        self.chk = None
        self.ray = None
        self.x = None

    def load_chk(self, step):
        """Load checkpoint data file
        """
        f_name = f'{self.chk_name}_hdf5_chk_{step:04d}'
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

