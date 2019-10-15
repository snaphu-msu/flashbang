import os
import numpy as np
import matplotlib.pyplot as plt
import yt


class BangSim:
    def __init__(self, name, bang_path='~/projects/codes/BANG/runs', xmax=1e12,
                 dim=1):
        self.path = os.path.join(bang_path, f'run_{name}')
        self.name = name
        self.dim = dim
        self.xmax = xmax
