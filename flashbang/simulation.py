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
#   - generalised axis plotting
#       - save/show plot


# noinspection PyTypeChecker
class Simulation:
    def __init__(self, model, basename='run', runs_path=None, config='default',
                 xmax=1e12, output_dir='output', verbose=True,
                 load_all=True, reload=False, save=True,
                 trans_dens=6e7, trans_low=1e7, runs_prefix='run_'):
        """Object representing a 1D flash simulation

        parameters
        ----------
        model : str
            The label for the model directory, e.g. 'helmNet' for 'run_helmNet/'
        basename : str
            The label that's used in chk and .dat files, e.g. 'run' for 'run.dat'
        runs_path : str
            Override the default place to look for models (bash variable: BANG_MODELS)
        runs_prefix : str
            The prefix added to 'model', e.g. 'run_' for 'run_helmNet'
        config : str
            Base name of config file to use, e.g. 'default' for 'config/default.ini'
        xmax : float
        output_dir : str
            name of subdirectory containing model output files
        load_all : bool
            immediately load all model data (chk profiles, dat)
        reload : bool
            force reload model data from raw files (don't load from temp/)
        save : bool
            save extracted model data to temporary files (for faster loading)
        trans_dens : float
            helmholtz transition density (hybridEOS only)
        trans_low : float
            helmholtz low transition density (hybridEOS only)
        verbose : bool
        """
        self.verbose = verbose
        self.runs_path = runs_path
        self.path = paths.model_path(model=model, runs_path=runs_path, runs_prefix=runs_prefix)
        self.output_path = os.path.join(self.path, output_dir)

        self.model = model
        self.basename = basename
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
        self.trans_low_idxs = np.full(self.n_chk, -1)
        self.trans_r = np.full(self.n_chk, np.nan)
        self.trans_low_r = np.full(self.n_chk, np.nan)

        if load_all:
            self.load_dat(reload=reload, save=save)
            self.load_all_profiles(reload=reload, save=save)
            self.find_trans_idxs()
            self.get_trans_r()

    def printv(self, string, verbose=None):
        """Verbose-aware print

        parameters
        ----------
        string : str
            string to print if verbose=True
        verbose : bool
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print(string)

    def load_dat(self, reload=False, save=True):
        """Load .dat file

        parameters
        ----------
        reload : bool
        save : bool
        """
        self.dat = load_save.get_dat(self.basename, model=self.model, runs_path=self.runs_path,
                                     cols_dict=self.config['dat_columns'], reload=reload,
                                     save=save)

    def update_chk_list(self):
        """Update the list of checkpoint files available
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
            checkpoint file to load from (e.g., chk=12 for `run_hdf5_chk_0012`
        reload : bool
        save : bool
        """
        config = self.config['profile']
        params = config['params'] + config['composition']

        self.profiles[chk] = load_save.get_profile(
                            self.basename, chk=chk, model=self.model, xmax=self.xmax,
                            o_path=self.output_path, params=params, reload=reload,
                            save=save, verbose=self.verbose)

    def find_trans_idxs(self):
        """Find idxs for zones closest to the helmholtz transition densities
        for each chk profile
        """
        self.printv('Finding helmholtz transition zones')

        for i, chk in enumerate(self.chk_list):
            profile = self.profiles[chk]
            dens_reverse = np.flip(profile['dens'])  # need monotonically-increasing

            for j, trans in enumerate([self.trans_dens, self.trans_low]):
                idx_list = [self.trans_idxs, self.trans_low_idxs][j]
                trans_idx = tools.find_nearest_idx(dens_reverse, trans)

                max_idx = len(dens_reverse) - 1
                idx_list[i] = max_idx - trans_idx  # flip back

    def get_trans_r(self):
        """Get radii at transition zones
        """
        if np.any(self.trans_idxs < 0) or np.any(self.trans_low_idxs < 0):
            self.find_trans_idxs()

        self.printv('Getting transition zone radii')

        for i, trans_idx in enumerate(self.trans_idxs):
            chk = self.chk_list[i]
            profile = self.profiles[chk]
            self.trans_r[i] = profile['x'][trans_idx]
            self.trans_low_r[i] = profile['x'][self.trans_low_idxs[i]]

    def plot_profiles(self, chk, y_var_list, x_var='x', y_scale=None, x_scale=None,
                      max_cols=2, sub_figsize=(6, 5), trans=True, legend=False):
        """Plot one or more profile variables

        parameters
        ----------
        chk : int
            checkpoint to plot
        y_var_list : str | [str]
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
        y_var_list = tools.ensure_sequence(y_var_list)
        n_var = len(y_var_list)
        fig, ax = plot_tools.setup_subplots(n_var, max_cols=max_cols,
                                            sub_figsize=sub_figsize, squeeze=False)

        for i, y_var in enumerate(y_var_list):
            row = int(np.floor(i / max_cols))
            col = i % max_cols

            self.plot_profile(chk, y_var=y_var, x_var=x_var, y_scale=y_scale,
                              x_scale=x_scale, ax=ax[row, col], trans=trans,
                              legend=legend if i == 0 else False)
        return fig

    def plot_profile(self, chk, y_var, x_var='x', y_scale=None, x_scale=None,
                     ax=None, legend=False, trans=True, title=True,
                     ylims=None, xlims=None, figsize=(8, 6), label=None):
        """Plot given profile variable

        parameters
        ----------
        chk : int | [int]
            checkpoint(s) to plot
        y_var : str
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
        label : str
        """
        chk = tools.ensure_sequence(chk)

        for i in chk:
            if i not in self.profiles.keys():
                self.load_profile(i)

        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        self._set_ax_title(ax, chk=chk[0], title=title)
        self._set_ax_scales(ax, y_var, x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_lims(ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(ax, x_var=x_var, y_var=y_var)

        for i in chk:
            profile = self.profiles[i]
            y = profile[y_var]

            ax.plot(profile[x_var], y, ls='-', marker='', label=label)
            self._plot_trans_line(x_var, y=y, ax=ax, chk=i, trans=trans)

        if legend:
            ax.legend()

        return fig

    def plot_composition(self, chk, x_var='x', y_scale='log', x_scale=None,
                         y_var_list=('neut', 'prot', 'he4', 'o16', 'si28', 'fe54', 'fe56'),
                         ax=None, legend=True, ylims=(1e-5, 2), xlims=None,
                         trans=True, figsize=(8, 6), title=True):
        """Plot isotope composition profile

        parameters
        ----------
        chk : int
        y_var_list : [str]
            list of isotopes to plot (see self.config['profile']['params'])
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
        if chk not in self.profiles.keys():
            self.load_profile(chk)

        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        self._set_ax_scales(ax, y_var_list[0], x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_title(ax, chk=chk, title=title)
        self._set_ax_lims(ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(ax, x_var=x_var, y_var='$X$')

        profile = self.profiles[chk]
        for key in y_var_list:
            ax.plot(profile[x_var], profile[key], label=f'{key}')

        self._plot_trans_line(x_var, y=ylims, ax=ax, chk=chk, trans=trans)

        if legend:
            ax.legend(loc='lower right')

        return fig

    def plot_slider(self, y_var, x_var='x', y_scale=None, x_scale=None, trans=True,
                    figsize=(8, 6), title=True, xlims=None, ylims=None):
        """Plot interactive slider of profile for given variable

        parameters
        ----------
        y_var : str
        x_var : str
        y_scale : str
        x_scale : str
        trans : bool
            plot helmholtz transitions
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
        line, = profile_ax.plot(init_profile[x_var], init_profile[y_var], ls='-', marker='')

        self._set_ax_scales(profile_ax, y_var, x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_title(profile_ax, chk=j_init, title=title)
        self._set_ax_lims(profile_ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(profile_ax, x_var=x_var, y_var=y_var)

        self._plot_trans_line(x_var=x_var, y=init_profile[y_var], ax=profile_ax,
                              chk=j_init, trans=trans)

        slider = Slider(slider_ax, 'chk', j_min, j_max, valinit=j_init, valstep=1)

        def update(chk):
            idx = int(chk)
            profile = self.profiles[idx]
            y_profile = profile[y_var]

            line.set_ydata(y_profile)
            line.set_xdata(profile[x_var])
            self._set_ax_title(profile_ax, chk=idx, title=title)

            if trans:
                # TODO: nicer way to do this
                x, y = self._get_trans_xy(chk=idx, x_var=x_var, y=y_profile)
                for i in range(2):
                    profile_ax.lines[i+1].set_xdata(x[i])
                    profile_ax.lines[i+1].set_ydata(y)

            fig.canvas.draw_idle()

        slider.on_changed(update)
        return fig, slider

    def plot_dat(self, y_var, y_scale='log', display=True, ax=None, figsize=(8, 6)):
        """Plot quantity from dat file

        parameters
        ----------
        y_var : str
        y_scale : str
        figsize : []
        display : bool
        ax : pyplot.axis
        """
        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        ax.plot(self.dat['time'], self.dat[y_var])

        ax.set_yscale(y_scale)
        self._set_ax_labels(ax, x_var='$t$ (s)', y_var=y_var)

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
        y : []
            1D array of y-values
        """
        idx = np.where(self.chk_list == chk)[0][0]
        y_max = np.max(y)
        y_min = np.min(y)

        y_min = 0  # TODO: automagic this
        y_max = 20

        # TODO: nicer way to handle both dens and low
        x_map = {
                 'dens': [self.trans_dens, self.trans_low],
                 'x': [self.trans_r[idx], self.trans_low_r[idx]],
                 }.get(x_var)

        x = [[x_map[0], x_map[0]], [x_map[1], x_map[1]]]
        y = [y_min, y_max]
        return x, y

    def _plot_trans_line(self, x_var, y, ax, chk, trans):
        """Add transition line to axis

        parameters
        ----------
        x_var : str
        y : []
            1D array of y-values
        ax : pyplot.axis
        chk : int
        trans : bool
        """
        if trans:
            x, y = self._get_trans_xy(chk=chk, x_var=x_var, y=y)
            for i in range(2):
                ax.plot(x[i], y, ls='--', color='k')

    def _set_ax_scales(self, ax, y_var, x_var, y_scale, x_scale):
        """Set axis scales (linear, log)

        parameters
        ----------
        ax : pyplot.axis
        y_var : str
        x_var : str
        y_scale : str
        x_scale : str
        """
        if x_scale is None:
            x_scale = self.config['plotting']['ax_scales'].get(x_var, 'log')
        if y_scale is None:
            y_scale = self.config['plotting']['ax_scales'].get(y_var, 'log')

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

    def _set_ax_title(self, ax, chk, title):
        """Set axis title

        parameters
        ----------
        ax : pyplot.axis
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
        ax : pyplot.axis
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
        ax : pyplot.axis
        x_var : str
        y_var : str
        """
        ax.set_xlabel(self.get_label(x_var))
        ax.set_ylabel(self.get_label(y_var))

    def _setup_fig_ax(self, ax, figsize):
        """Setup fig, ax, checking if ax already provided
        """
        c = self.config['plotting']  # TODO: default settings from config
        fig = None

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        return fig, ax
