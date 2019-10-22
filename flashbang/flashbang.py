import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import multiprocessing as mp

# bangpy
from . import paths
from . import load_save

# TODO:
#   - plotly slider of profiles


class Simulation:
    def __init__(self, model, basename=None, runs_path=None, config='default',
                 xmax=1e12, dim=1, output_dir='output', verbose=True,
                 load_all=True):
        self.verbose = verbose
        self.runs_path = runs_path
        self.path = paths.model_path(model=model, runs_path=runs_path)
        self.output_path = os.path.join(self.path, output_dir)

        self.model = model
        self.basename = basename
        self.dim = dim
        self.xmax = xmax

        self.config = load_save.load_config(name=config, verbose=self.verbose)
        self.dat = None
        self.profiles = {}

        self.chk_idxs = load_save.find_chk(path=self.output_path)

        if load_all:
            self.load_dat()

    def printv(self, string):
        if self.verbose:
            print(string)

    def load_dat(self):
        """Load .dat file
        """
        self.dat = load_save.load_dat(self.basename, model=self.model,
                                      cols_dict=self.config['dat_columns'])

    def load_all_profiles(self, reload=False, save=True, multithread=False):
        """Load profiles for all available checkpoints

        parameters
        ----------
        reload : bool
        save : bool
        multithread : bool
        """
        if multithread:
            args = []
            for chk_i in self.chk_idxs:
                args.append((chk_i, reload, save))

            with mp.Pool(processes=4) as pool:
                pool.starmap(self.load_profile, args)
        else:
            for chk_i in self.chk_idxs:
                self.load_profile(chk_i, reload=reload, save=save)

    def load_profile(self, chk_i, reload=False, save=True):
        """Load checkpoint data file

        parameters
        ----------
        chk_i : int
            checkpoint ID to load
        reload : bool
        save : bool
        """
        self.profiles[chk_i] = load_save.extract_profile(
                                    self.basename, chk_i=chk_i, model=self.model,
                                    xmax=self.xmax, o_path=self.output_path,
                                    params=self.config['profile']['params'],
                                    reload=reload, save=save, verbose=self.verbose)

    def get_label(self, key):
        """Return formatted string for plot label
        """
        return self.config['plotting']['labels'].get(key, key)

    def plot_profile(self, chk_i, var, y_log=True, x_log=True):
        """Plot given profile variable

        parameters
        ----------
        chk_i : int
            checkpoint ID to plot
        var : str
            variable to plot on y-axis (from Simulation.profile)
        y_log : bool
        x_log : bool
        """
        try:
            profile = self.profiles[chk_i]
        except KeyError:
            self.load_profile(chk_i)
            profile = self.profiles[chk_i]

        y_profile = profile[var]

        fig, ax = plt.subplots()
        ax.plot(profile['x'], y_profile)

        if y_log:
            ax.set_yscale('log')
        if x_log:
            ax.set_xscale('log')

        ax.set_ylabel(self.get_label(var))
        ax.set_xlabel(self.get_label('x'))

    def plot_slider(self, var, y_log=True, x_log=True):
        """Plot interactive slider of profile for given variable

        parameters
        ----------
        var : str
        y_log : bool
        x_log : bool
        """
        a_min = 0  # the minimial value of the paramater a
        a_max = self.chk_idxs[-1]  # the maximal value of the paramater a
        a_init = a_max

        fig = plt.figure(figsize=(8, 4))

        x = np.linspace(0, 2 * np.pi, 500)

        profile_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
        plt.axes(profile_ax)  # select sin_ax
        sin_plot, = plt.plot(self.profiles[a_init]['x'], self.profiles[a_init][var])

        plt.ylabel(self.get_label(var))
        plt.xlabel(self.get_label('x'))

        if y_log:
            plt.yscale('log')
        if x_log:
            plt.xscale('log')

        a_slider = Slider(slider_ax, 'chk', a_min, a_max, valinit=a_init, valstep=1)

        def update(chk):
            profile = self.profiles[chk]
            y_profile = profile[var]
            sin_plot.set_ydata(y_profile)
            sin_plot.set_xdata(profile['x'])
            fig.canvas.draw_idle()  # redraw the plot

        a_slider.on_changed(update)

        # plt.show()
        return a_slider

    def plot_dat(self, var, y_log=True, display=True):
        """Plots quantity from dat file
        """
        fig, ax = plt.subplots()
        if y_log:
            ax.set_yscale('log')

        ax.plot(self.dat['time'], self.dat[var])

        ax.set_xlabel('$t$ (s)')
        ax.set_ylabel(var)

        if display:
            plt.show(block=False)
