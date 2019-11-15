import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# bangpy
from . import paths
from . import load_save
from . import tools
from . import plot_tools

# TODO:
#   - add docstring parameters
#   - rename chk_i to chk, use chk_i for actual index
#   - generalised axis plotting


# noinspection PyTypeChecker
class Simulation:
    def __init__(self, basename, model, runs_path=None, config='default',
                 xmax=1e12, dim=1, output_dir='output', verbose=True,
                 load_dat=False, load_profiles=False, reload=False,
                 trans_dens=6e7):
        """Represents a 1D flash simulation
        """
        self.verbose = verbose
        self.runs_path = runs_path
        self.path = paths.model_path(model=model, runs_path=runs_path)
        self.output_path = os.path.join(self.path, output_dir)

        self.model = model
        self.basename = basename
        self.dim = dim
        self.xmax = xmax
        self.trans_dens = trans_dens

        self.config = load_save.load_config(name=config, verbose=self.verbose)
        self.dat = None
        self.chk_idxs = None
        self.profiles = {}

        self.update_chks()
        self.n_chk = len(self.chk_idxs)
        self.trans_idxs = np.full(self.n_chk, -1)
        self.trans_r = np.full(self.n_chk, np.nan)

        if load_dat:
            self.load_dat()
        if load_profiles:
            self.load_all_profiles(reload=reload)
            self.find_trans_idxs()
            self.get_trans_r()

    def printv(self, string, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print(string)

    def load_dat(self):
        """Load .dat file
        """
        self.dat = load_save.load_dat(self.basename, model=self.model,
                                      cols_dict=self.config['dat_columns'])

    def update_chks(self):
        """Update the checkpoint files available
        """
        self.chk_idxs = load_save.find_chk(path=self.output_path,
                                           match_str=f'{self.basename}_hdf5_chk_')

    def load_all_profiles(self, reload=False, save=True):
        """Load profiles for all available checkpoints

        parameters
        ----------
        reload : bool
        save : bool
        """
        self.printv(f'Loading chk profiles from: {self.output_path}')

        verbose_setting = self.verbose  # verbosity hack
        self.verbose = False

        for chk_i in self.chk_idxs:
            if verbose_setting:
                sys.stdout.write(f'\rchk: {chk_i}/{self.chk_idxs[-1]}')
            self.load_profile(chk_i, reload=reload, save=save)

        if verbose_setting:
            sys.stdout.write('\n')
        self.verbose = verbose_setting

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

    def find_trans_idxs(self):
        """Finds idx for zone closest to the helmholtz transition density,
                for each chk profile
        """
        self.printv('Finding helmholtz transition zones')

        for i, chk in enumerate(self.chk_idxs):
            profile = self.profiles[chk]
            dens_reverse = np.flip(profile['dens'])  # need monotonically-increasing
            trans_idx = tools.find_nearest_idx(dens_reverse, self.trans_dens)

            max_idx = len(dens_reverse) - 1
            self.trans_idxs[i] = max_idx - trans_idx  # flip back

    def get_trans_r(self):
        """Gets radii at transition zones
        """
        if np.any(self.trans_idxs < 0):
            self.find_trans_idxs()

        self.printv('Getting transition zone radii')
        for i, trans_idx in enumerate(self.trans_idxs):
            chk_i = self.chk_idxs[i]
            profile = self.profiles[chk_i]
            self.trans_r[i] = profile['x'][trans_idx]

    def get_label(self, key):
        """Return formatted string for plot label
        """
        return self.config['plotting']['labels'].get(key, key)

    def plot_profiles(self, chk_i, var_list, x_var='x', y_log=True, x_log=True,
                      max_cols=2, sub_figsize=(6, 5), trans=True):
        """Plot one or more profile variables

        parameters
        ----------
        chk_i : int
            checkpoint ID to plot
        var_list : str | [str]
            variable(s) to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_log : bool
        x_log : bool
        max_cols : bool
        sub_figsize : tuple
        trans : bool
        """
        chk_i = tools.ensure_sequence(chk_i)
        var_list = tools.ensure_sequence(var_list)
        n_var = len(var_list)
        fig, ax = plot_tools.setup_subplots(n_var, max_cols=max_cols,
                                            sub_figsize=sub_figsize, squeeze=False)

        for i, var in enumerate(var_list):
            row = int(np.floor(i / max_cols))
            col = i % max_cols

            self.plot_profile(chk_i, var=var, x_var=x_var, y_log=y_log, x_log=x_log,
                              ax=ax[row, col], legend=True if i == 0 else False,
                              trans=trans)
        return fig

    def _get_trans_xy(self, chk_i, x_var, y):
        """Return x, y points of transition line, for given x-axis variable

        parameters
        ----------
        chk_i : int
        x_var : str
        y     : []
        """
        idx = np.where(self.chk_idxs == chk_i)[0][0]
        y_max = np.max(y)
        y_min = np.min(y)

        x_map = {
                 'dens': self.trans_dens,
                 'x': self.trans_r[idx]
                 }.get(x_var)

        x = [x_map, x_map]
        y = [y_min, y_max]
        return x, y

    def _plot_trans_line(self, x_var, y, ax, chk_i, trans):
        """Add transition line to axis

        parameters
        ----------
        x_var : str
            variable on x-axis
        y : []
            array of y-axis values
        ax : plt.axis
            pyplot axis to plot on
        chk_i : int
            checkpoint index
        trans : bool
            whether to plot transition line
        """
        if trans:
            x, y = self._get_trans_xy(chk_i=chk_i, x_var=x_var, y=y)
            ax.plot(x, y, ls='--', color='k')

    def plot_profile(self, chk_i, var, x_var='x', yscale=None, x_log=True,
                     ax=None, legend=True, trans=True, title=True,
                     ylims=None, xlims=None, figsize=(8, 6)):
        """Plot given profile variable

        parameters
        ----------
        chk_i : int | [int]
            checkpoint(s) to plot
        var : str
            variable to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        yscale : str
        x_log : bool
        ax : pyplot.axis
        legend : bool
        trans : bool
        title : bool
        ylims : []
        xlims : []
        figsize : []
        """
        chk_i = tools.ensure_sequence(chk_i)

        for i in chk_i:
            if i not in self.profiles.keys():
                self.load_profile(i)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for i in chk_i:
            profile = self.profiles[i]
            y = profile[var]

            ax.plot(profile[x_var], y, label=f'{i}')
            self._plot_trans_line(x_var, y=y, ax=ax, chk_i=i, trans=trans)

        if yscale is None:
            yscale = self.config['plotting']['y_scales'].get(var, 'log')
        if x_log:
            ax.set_xscale('log')
        if legend:
            ax.legend()
        if title:   # TODO: account for different zero points/starting times
            dt = self.config['plotting']['scales']['chk_dt']
            time = dt * chk_i[0]
            ax.set_title(f't={time:.3f} s')

        ax.set_yscale(yscale)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        ax.set_ylabel(self.get_label(var))
        ax.set_xlabel(self.get_label(x_var))

        return fig

    def plot_composition(self, chk_i, var_list=('neut', 'prot', 'si28', 'fe54', 'fe56'),
                         x_var='x', y_log=True, x_log=True, ax=None, legend=True,
                         ylims=(1e-5, 2), trans=True, figsize=(8, 6)):
        """Plots composition profile
        """
        if chk_i not in self.profiles.keys():
            self.load_profile(chk_i)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylim(ylims)

        profile = self.profiles[chk_i]
        for key in var_list:
            ax.plot(profile[x_var], profile[key], label=f'{key}')

        self._plot_trans_line(x_var, y=ylims, ax=ax, chk_i=chk_i, trans=trans)

        if y_log:
            ax.set_yscale('log')
        if x_log:
            ax.set_xscale('log')
        if legend:
            ax.legend()

        ax.set_ylabel('$X$')
        ax.set_xlabel(self.get_label(x_var))

    def plot_slider(self, var, x_var='x', y_log=True, x_log=True, trans=True,
                    figsize=(8, 6)):
        """Plot interactive slider of profile for given variable

        parameters
        ----------
        var : str
        x_var : str
        y_log : bool
        x_log : bool
        trans : bool
        figsize : []
        """
        j_max = self.chk_idxs[-1]
        j_min = self.chk_idxs[0]
        j_init = j_max

        fig = plt.figure(figsize=figsize)
        profile_ax = fig.add_axes([0.1, 0.2, 0.8, 0.65])
        slider_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05])
        profile_ax.set_xlabel(self.get_label(x_var))
        profile_ax.set_ylabel(self.get_label(var))

        init_profile = self.profiles[j_init]
        line, = profile_ax.plot(init_profile[x_var], init_profile[var])

        self._plot_trans_line(x_var=x_var, y=init_profile[var], ax=profile_ax,
                              chk_i=j_init, trans=trans)

        if y_log:
            profile_ax.set_yscale('log')
        if x_log:
            profile_ax.set_xscale('log')

        slider = Slider(slider_ax, 'chk', j_min, j_max, valinit=j_init, valstep=1)

        def update(chk):
            idx = int(chk)
            profile = self.profiles[idx]
            y_profile = profile[var]

            line.set_ydata(y_profile)
            line.set_xdata(profile[x_var])

            if trans:
                x, y = self._get_trans_xy(chk_i=idx, x_var=x_var, y=y_profile)
                profile_ax.lines[1].set_xdata(x)
                profile_ax.lines[1].set_ydata(y)

            fig.canvas.draw_idle()

        slider.on_changed(update)
        return fig, slider

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
