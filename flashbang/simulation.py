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
#   - rename var to y_var
#   - generalised axis plotting
#       - save/show plot
#   - add attr:
#       - save


# noinspection PyTypeChecker
class Simulation:
    def __init__(self, basename, model, runs_path=None, config='default',
                 xmax=1e12, dim=1, output_dir='output', verbose=True,
                 load_dat=True, load_profiles=False, reload=False,
                 trans_dens=6e7, trans_low=1e7, runs_prefix='run_'):
        """Represents a 1D flash simulation
        """
        # TODO: docstring parameters
        self.verbose = verbose
        self.runs_path = runs_path
        self.path = paths.model_path(model=model, runs_path=runs_path, runs_prefix=runs_prefix)
        self.output_path = os.path.join(self.path, output_dir)

        self.model = model
        self.basename = basename
        self.dim = dim
        self.xmax = xmax
        self.trans_dens = trans_dens
        self.trans_low = trans_low

        self.config = load_save.load_config(name=config, verbose=self.verbose)
        self.dat = None
        self.chk_list = None
        self.profiles = {}

        self.update_chk_list()
        self.n_chk = len(self.chk_list)
        self.trans_idxs = np.full(self.n_chk, -1)
        self.trans_r = np.full(self.n_chk, np.nan)

        if load_dat:
            self.load_dat(reload=reload)
        if load_profiles:
            self.load_all_profiles(reload=reload)
            self.find_trans_idxs()
            self.get_trans_r()

    def printv(self, string, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print(string)

    def load_dat(self, reload=False, save=True):
        """Load .dat file
        """
        self.dat = load_save.get_dat(self.basename, model=self.model, runs_path=self.runs_path,
                                     cols_dict=self.config['dat_columns'], reload=reload,
                                     save=save)

    def update_chk_list(self):
        """Update the checkpoint files available
        """
        self.chk_list = load_save.find_chk(path=self.output_path,
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

        for chk in self.chk_list:
            if verbose_setting:
                sys.stdout.write(f'\rchk: {chk}/{self.chk_list[-1]}')
            self.load_profile(chk, reload=reload, save=save)

        if verbose_setting:
            sys.stdout.write('\n')
        self.verbose = verbose_setting

    def load_profile(self, chk, reload=False, save=True):
        """Load checkpoint data file

        parameters
        ----------
        chk : int
            checkpoint ID to load
        reload : bool
        save : bool
        """
        self.profiles[chk] = load_save.get_profile(
                                    self.basename, chk=chk, model=self.model,
                                    xmax=self.xmax, o_path=self.output_path,
                                    params=self.config['profile']['params'],
                                    reload=reload, save=save, verbose=self.verbose)

    def find_trans_idxs(self):
        """Finds idx for zone closest to the helmholtz transition density,
                for each chk profile
        """
        self.printv('Finding helmholtz transition zones')

        for i, chk in enumerate(self.chk_list):
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
            chk = self.chk_list[i]
            profile = self.profiles[chk]
            self.trans_r[i] = profile['x'][trans_idx]

    def plot_profiles(self, chk, var_list, x_var='x', y_scale=None, x_scale=None,
                      max_cols=2, sub_figsize=(6, 5), trans=True, legend=False):
        """Plot one or more profile variables

        parameters
        ----------
        chk : int
            checkpoint ID to plot
        var_list : str | [str]
            variable(s) to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_scale : bool
        x_scale : bool
        legend : bool
        max_cols : bool
        sub_figsize : tuple
        trans : bool
        """
        chk = tools.ensure_sequence(chk)
        var_list = tools.ensure_sequence(var_list)
        n_var = len(var_list)
        fig, ax = plot_tools.setup_subplots(n_var, max_cols=max_cols,
                                            sub_figsize=sub_figsize, squeeze=False)

        for i, var in enumerate(var_list):
            row = int(np.floor(i / max_cols))
            col = i % max_cols

            self.plot_profile(chk, var=var, x_var=x_var, y_scale=y_scale,
                              x_scale=x_scale, ax=ax[row, col], trans=trans,
                              legend=legend if i == 0 else False)
        return fig

    def plot_profile(self, chk, var, x_var='x', y_scale=None, x_scale=None,
                     ax=None, legend=False, trans=True, title=True,
                     ylims=None, xlims=None, figsize=(8, 6)):
        """Plot given profile variable

        parameters
        ----------
        chk : int | [int]
            checkpoint(s) to plot
        var : str
            variable to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_scale : str
        x_scale : bool
        ax : pyplot.axis
        legend : bool
        trans : bool
        title : bool
        ylims : []
        xlims : []
        figsize : []
        """
        chk = tools.ensure_sequence(chk)

        for i in chk:
            if i not in self.profiles.keys():
                self.load_profile(i)

        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        self._set_ax_title(ax, chk=chk[0], title=title)
        self._set_ax_scales(ax, var, x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_lims(ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(ax, x_var=x_var, y_var=var)

        for i in chk:
            profile = self.profiles[i]
            y = profile[var]

            ax.plot(profile[x_var], y, ls='-', marker='', label=f'{i}')
            self._plot_trans_line(x_var, y=y, ax=ax, chk=i, trans=trans)

        if legend:
            ax.legend()

        return fig

    def plot_composition(self, chk, var_list=('neut', 'prot', 'si28', 'fe54', 'fe56'),
                         x_var='x', y_scale='log', x_scale=None, ax=None, legend=True,
                         ylims=(1e-5, 2), xlims=None, trans=True, figsize=(8, 6),
                         title=True):
        """Plots composition profile
        """
        if chk not in self.profiles.keys():
            self.load_profile(chk)

        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        self._set_ax_scales(ax, var_list[0], x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_title(ax, chk=chk, title=title)
        self._set_ax_lims(ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(ax, x_var=x_var, y_var='$X$')

        profile = self.profiles[chk]
        for key in var_list:
            ax.plot(profile[x_var], profile[key], label=f'{key}')

        self._plot_trans_line(x_var, y=ylims, ax=ax, chk=chk, trans=trans)

        if legend:
            ax.legend()

        return fig

    def plot_slider(self, var, x_var='x', y_scale=None, x_scale=None, trans=True,
                    figsize=(8, 6), title=True, xlims=None, ylims=None):
        """Plot interactive slider of profile for given variable

        parameters
        ----------
        var : str
        x_var : str
        y_scale : str
        x_scale : str
        trans : bool
        figsize : []
        title : bool
        xlims : [2]
        ylims : [2]
        """
        j_max = self.chk_list[-1]
        j_min = self.chk_list[0]
        j_init = j_max

        fig = plt.figure(figsize=figsize)
        profile_ax = fig.add_axes([0.1, 0.2, 0.8, 0.65])
        slider_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05])

        init_profile = self.profiles[j_init]
        line, = profile_ax.plot(init_profile[x_var], init_profile[var], ls='-', marker='')

        self._set_ax_scales(profile_ax, var, x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_title(profile_ax, chk=j_init, title=title)
        self._set_ax_lims(profile_ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(profile_ax, x_var=x_var, y_var=var)

        self._plot_trans_line(x_var=x_var, y=init_profile[var], ax=profile_ax,
                              chk=j_init, trans=trans)

        slider = Slider(slider_ax, 'chk', j_min, j_max, valinit=j_init, valstep=1)

        def update(chk):
            idx = int(chk)
            profile = self.profiles[idx]
            y_profile = profile[var]

            line.set_ydata(y_profile)
            line.set_xdata(profile[x_var])
            self._set_ax_title(profile_ax, chk=idx, title=title)

            if trans:
                x, y = self._get_trans_xy(chk=idx, x_var=x_var, y=y_profile)
                profile_ax.lines[1].set_xdata(x)
                profile_ax.lines[1].set_ydata(y)

            fig.canvas.draw_idle()

        slider.on_changed(update)
        return fig, slider

    def plot_dat(self, var, y_scale='log', display=True, ax=None, figsize=(8, 6)):
        """Plot quantity from dat file
        """
        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        ax.plot(self.dat['time'], self.dat[var])

        ax.set_yscale(y_scale)
        self._set_ax_labels(ax, x_var='$t$ (s)', y_var=var)

        if display:
            plt.show(block=False)

    def get_label(self, key):
        """Return formatted string for plot label
        """
        return self.config['plotting']['labels'].get(key, key)

    def _get_trans_xy(self, chk, x_var, y):
        """Return x, y points of transition line, for given x-axis variable

        parameters
        ----------
        chk : int
        x_var : str
        y     : []
        """
        idx = np.where(self.chk_list == chk)[0][0]
        y_max = np.max(y)
        y_min = np.min(y)

        # y_min = 0  # TODO: automagic this
        # y_max = 20

        x_map = {
                 'dens': self.trans_dens,
                 'x': self.trans_r[idx]
                 }.get(x_var)

        x = [x_map, x_map]
        y = [y_min, y_max]
        return x, y

    def _plot_trans_line(self, x_var, y, ax, chk, trans):
        """Add transition line to axis

        parameters
        ----------
        x_var : str
            variable on x-axis
        y : []
            array of y-axis values
        ax : plt.axis
            pyplot axis to plot on
        chk : int
            checkpoint index
        trans : bool
            whether to plot transition line
        """
        if trans:
            x, y = self._get_trans_xy(chk=chk, x_var=x_var, y=y)
            ax.plot(x, y, ls='--', color='k')

    def _set_ax_scales(self, ax, var, x_var, y_scale, x_scale):
        """Set axis scales (linear, log)

        parameters
        ----------
        ax : plt.axis
        var : str
        x_var : str
        y_scale : str
        x_scale : str
        """
        if x_scale is None:
            x_scale = self.config['plotting']['ax_scales'].get(x_var, 'log')
        if y_scale is None:
            y_scale = self.config['plotting']['ax_scales'].get(var, 'log')

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

    def _set_ax_title(self, ax, chk, title):
        """Set axis title

        parameters
        ----------
        ax : plt.axis
        chk : int
        title : bool
        """
        # TODO: account for different zero points/starting times
        if title:
            dt = self.config['plotting']['scales']['chk_dt']
            time = dt * chk
            ax.set_title(f't={time:.3f} s')

    def _set_ax_lims(self, ax, xlims, ylims):
        """Set x and y axis limits

        parameters
        ----------
        ax : plt.axis
        xlims : []
        ylims : []
        """
        c = self.config['plotting']  # TODO: something with auto-lims in future
        if ylims is not None:
            ax.set_ylim(ylims)
        if xlims is not None:
            ax.set_xlim(xlims)

    def _set_ax_labels(self, ax, x_var, y_var):
        """Set axis labels

        parameters
        ----------
        ax : plt.axis
        x_var : str
        y_var : str
        """
        ax.set_xlabel(self.get_label(x_var))
        ax.set_ylabel(self.get_label(y_var))

    def _setup_fig_ax(self, ax, figsize):
        """Sets up fig, ax
        """
        c = self.config['plotting']  # TODO: default settings from config
        fig = None

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        return fig, ax
