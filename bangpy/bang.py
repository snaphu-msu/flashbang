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

    def load_chk(self, step):
        """Loads checkpoint data file
        """
        f_name = f'{self.chk_name}_hdf5_chk_{step:04d}'
        f_path = os.path.join(self.output_path, f_name)
        self.chk = yt.load(f_path)
        self.ray = self.chk.ray([0, 0, 0], [self.xmax, 0, 0])