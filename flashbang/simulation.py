import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# flashbang
from . import analysis
from . import load_save
from . import paths
from . import plot_tools
from . import tools

# TODO:
#   - load progenitor model
#   - create an rcparams for default values of run, etc.
#   - generalised axis plotting
#       - save/show plot

# TODO: chk_table, add columns:
#   - time
#   - n_step
#   - n_zones


# noinspection PyTypeChecker
class Simulation:
    def __init__(self, model, run='run', runs_path=None, config='default',
                 output_dir='output', verbose=True,
                 load_all=True, reload=False, save=True,
                 runs_prefix='run_', trans=None):
        """Object representing a 1D flash simulation

        parameters
        ----------
        model : str
            The label for the model directory, e.g. 'helmNet' for 'run_helmNet/'
        run : str
            The label that's used in chk and .dat files, e.g. 'run' for 'run.dat'
        runs_path : str
            Override the default place to look for models (bash variable: BANG_MODELS)
        runs_prefix : str
            The prefix added to 'model', e.g. 'run_' for 'run_helmNet'
        config : str
            Base name of config file to use, e.g. 'default' for 'config/default.ini'
        output_dir : str
            name of subdirectory containing model output files
        load_all : bool
            immediately load all model data (chk profiles, dat)
        reload : bool
            force reload model data from raw files (don't load from temp/)
        save : bool
            save extracted model data to temporary files (for faster loading)
        verbose : bool
        trans : {}
            transition densities to track (g/cm^3), e.g. {'': 6e7, 'low': 1e7}
        """
        t0 = time.time()
        self.verbose = verbose
        self.runs_path = runs_path
        self.path = paths.model_path(model=model, runs_path=runs_path, runs_prefix=runs_prefix)
        self.output_path = os.path.join(self.path, output_dir)

        self.model = model
        self.run = run

        self.config = load_save.load_config(name=config, verbose=self.verbose)
        self.chk_table = pd.DataFrame()
        self.dat = None
        self.bounce_time = None
        self.profiles = {}
        self.tracers = {}

        if trans is None:
            trans = self.config['transitions']['dens']
        self.trans = trans

        self.n_chk = None
        self.update_chk_list()

        if load_all:
            self.load_all(reload=reload, save=save)

        t1 = time.time()
        self.printv(f'Total load time: {t1-t0:.3f} s')

    # =======================================================
    #                      Setup/init
    # =======================================================
    def printv(self, string, verbose=None, **kwargs):
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
            print(string, **kwargs)

    def load_all(self, reload, save):
        """Load all model data
        """
        self.get_bounce_time()
        self.load_dat(reload=reload, save=save)
        self.load_all_profiles(reload=reload, save=save)

        if self.trans is not None:
            self.find_trans_idxs()

    def get_bounce_time(self):
        """Get bounce time (s) from log file
        """
        self.bounce_time = load_save.get_bounce_time(self.model, run=self.run,
                                                     runs_path=self.runs_path,
                                                     verbose=self.verbose)

    def update_chk_list(self):
        """Update the list of checkpoint files available
        """
        self.chk_table['chk'] = load_save.find_chk(path=self.output_path,
                                                   match_str=f'{self.run}_hdf5_chk_')
        self.chk_table.set_index('chk', inplace=True)
        self.n_chk = len(self.chk_table)

    # =======================================================
    #                   Loading Data
    # =======================================================
    def load_dat(self, reload=False, save=True):
        """Load .dat file

        parameters
        ----------
        reload : bool
        save : bool
        """
        self.dat = load_save.get_dat(
                        model=self.model, run=self.run, runs_path=self.runs_path,
                        cols_dict=self.config['dat_columns'], reload=reload, save=save)

    def load_all_profiles(self, reload=False, save=True):
        """Load profiles for all available checkpoints

        parameters
        ----------
        reload : bool
        save : bool
        """
        self.printv(f'Loading chk profiles from: {self.output_path}')
        chk_max = self.chk_table.index[-1]

        for chk in self.chk_table.index:
            self.printv(f'\rchk: {chk}/{chk_max}', end='')
            self.load_profile(chk, reload=reload, save=save, verbose=False)

        self.printv('')

    def load_profile(self, chk, reload=False, save=True, verbose=None):
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

        if verbose is None:
            verbose = self.verbose

        self.profiles[chk] = load_save.get_profile(
                                chk, model=self.model, run=self.run,
                                o_path=self.output_path, params=params,
                                derived_params=config['derived_params'],
                                reload=reload, save=save, verbose=verbose)

    # =======================================================
    #                 Analysis & Postprocessing
    # =======================================================
    def find_trans_idxs(self):
        """Find idxs for zones closest to the helmholtz transition densities
        for each chk profile
        """
        self.printv('Finding transition zones')

        for key, trans_dens in self.trans.items():
            idx_list = np.zeros(self.n_chk, dtype=int)

            for i, chk in enumerate(self.chk_table.index):
                profile = self.profiles[chk]
                dens_reverse = np.flip(profile['dens'])  # need monotonically-increasing
                max_idx = len(dens_reverse) - 1

                trans_idx = tools.find_nearest_idx(dens_reverse, trans_dens)
                idx_list[i] = max_idx - trans_idx  # flip back

            self.chk_table[f'{key}_i'] = idx_list

    def extract_tracers(self):
        """Construct mass tracers from profile data
        """
        # TODO:
        #   - include chk timesteps
        params = self.config['tracers']['params']
        mass_def = self.config['tracers']['mass_grid']
        mass_grid = np.linspace(mass_def[0], mass_def[1], mass_def[2])

        data_cube = analysis.extract_multi_mass_tracers(mass_grid,
                                                        profiles=self.profiles,
                                                        params=params,
                                                        verbose=self.verbose)
        self.tracers['mass_grid'] = mass_grid

        for i, mass in enumerate(mass_grid):
            self.tracers[i] = pd.DataFrame(index=self.chk_table.index,
                                           data=data_cube[:, i, :],
                                           columns=params)

    # =======================================================
    #                      Plotting
    # =======================================================
    def plot_profiles(self, chk, y_var_list, x_var='r', y_scale=None, x_scale=None,
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

    def plot_profile(self, chk, y_var, x_var='r', y_scale=None, x_scale=None,
                     ax=None, legend=False, trans=True, title=True,
                     ylims=None, xlims=None, figsize=(8, 6), label=None,
                     linestyle='-', marker=''):
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
        linestyle : str
        marker : str
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

            ax.plot(profile[x_var], y, ls=linestyle, marker=marker, label=label)
            self._plot_trans_line(x_var, y=y, ax=ax, chk=i, trans=trans)

        if legend:
            ax.legend()

        return fig

    def plot_composition(self, chk, x_var='r', y_scale='log', x_scale=None,
                         y_var_list=None, ax=None, legend=True, ylims=(1e-5, 2), xlims=None,
                         trans=True, figsize=(8, 6), title=True, legend_loc='lower left',
                         show_ye=True):
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
        legend_loc : str
        show_ye : bool
        """
        if chk not in self.profiles.keys():
            self.load_profile(chk)
        if y_var_list is None:
            y_var_list = self.config['plotting']['isotopes']

        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        self._set_ax_scales(ax, y_var_list[0], x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_title(ax, chk=chk, title=title)
        self._set_ax_lims(ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(ax, x_var=x_var, y_var='$X$')

        profile = self.profiles[chk]
        for key in y_var_list:
            ax.plot(profile[x_var], profile[key], label=f'{key}')
        if show_ye:
            ax.plot(profile[x_var], profile['ye'], '--', label=f'Ye', color='k')

        self._plot_trans_line(x_var, y=ylims, ax=ax, chk=chk, trans=trans)

        if legend:
            ax.legend(loc=legend_loc)

        return fig

    def plot_slider(self, y_var, x_var='r', y_scale=None, x_scale=None, trans=True,
                    figsize=(8, 6), title=True, xlims=None, ylims=None, legend=True,
                    linestyle='-', marker=''):
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
        legend : bool
        linestyle : str
        marker : str
        """
        fig, profile_ax, slider_ax = self._setup_slider_fig(figsize=figsize)
        chk_max, chk_min, chk_init = self._get_slider_chk()

        slider = Slider(slider_ax, 'chk', chk_min, chk_max, valinit=chk_init, valstep=1)

        self.plot_profile(chk_init, y_var=y_var, x_var=x_var, y_scale=y_scale,
                          x_scale=x_scale, ax=profile_ax, legend=legend, trans=trans,
                          title=title, ylims=ylims, xlims=xlims, figsize=figsize,
                          linestyle=linestyle, marker=marker)

        def update(chk):
            idx = int(chk)
            profile = self.profiles[idx]
            y_profile = profile[y_var]

            profile_ax.lines[0].set_ydata(y_profile)
            profile_ax.lines[0].set_xdata(profile[x_var])
            self._set_ax_title(profile_ax, chk=idx, title=title)

            if trans:
                for i, key in enumerate(self.trans):
                    x, y = self._get_trans_xy(chk=idx, key=key, x_var=x_var, y=y_profile)
                    profile_ax.lines[i+1].set_xdata(x)
                    profile_ax.lines[i+1].set_ydata(y)

            fig.canvas.draw_idle()

        slider.on_changed(update)
        return fig, slider

    def plot_slider_composition(self, y_var_list=None, x_var='r', y_scale=None, x_scale=None,
                                trans=True, figsize=(8, 6), title=True, xlims=None,
                                ylims=(1e-5, 2), legend=True, show_ye=True):
        """Plot interactive slider of isotope composition

        parameters
        ----------
        y_var_list : [str]
        x_var : str
        y_scale : str
        x_scale : str
        trans : bool
            plot helmholtz transitions
        figsize : []
        title : bool
        xlims : [2]
        ylims : [2]
        legend : bool
        show_ye : bool
        """
        # TODO:
        #   - create isotope palette
        fig, profile_ax, slider_ax = self._setup_slider_fig(figsize=figsize)
        chk_max, chk_min, chk_init = self._get_slider_chk()

        if y_var_list is None:
            y_var_list = self.config['plotting']['isotopes']

        slider = Slider(slider_ax, 'chk', chk_min, chk_max, valinit=chk_init, valstep=1)

        self.plot_composition(chk_init, x_var=x_var, y_scale=y_scale, x_scale=x_scale,
                              y_var_list=y_var_list, ax=profile_ax, legend=legend,
                              ylims=ylims, xlims=xlims, trans=trans, title=title,
                              show_ye=show_ye)

        def update(chk):
            idx = int(chk)
            profile = self.profiles[idx]

            if trans:
                # TODO: nicer way to do this
                for i, key in enumerate(self.trans):
                    x, y = self._get_trans_xy(chk=idx, key=key, x_var=x_var, y=ylims)
                    line_idx = -i - 1
                    profile_ax.lines[line_idx].set_xdata(x)
                    profile_ax.lines[line_idx].set_ydata(y)

            for i, key in enumerate(y_var_list):
                y_profile = profile[key]
                profile_ax.lines[i].set_xdata(profile[x_var])
                profile_ax.lines[i].set_ydata(y_profile)

            if show_ye:
                line_idx = len(y_var_list)
                profile_ax.lines[line_idx].set_xdata(profile[x_var])
                profile_ax.lines[line_idx].set_ydata(profile['ye'])

            self._set_ax_title(profile_ax, chk=idx, title=title)
            fig.canvas.draw_idle()

        slider.on_changed(update)
        return fig, slider

    def plot_dat(self, y_var, y_scale='log', display=True, ax=None, figsize=(8, 6),
                 linestyle='-', marker=''):
        """Plot quantity from dat file

        parameters
        ----------
        y_var : str
        y_scale : str
        figsize : []
        display : bool
        ax : pyplot.axis
        linestyle : str
        marker : str
        """
        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        ax.plot(self.dat['time'], self.dat[y_var], linestyle=linestyle, marker=marker)

        ax.set_yscale(y_scale)
        self._set_ax_labels(ax, x_var='$t$ (s)', y_var=y_var)

        if display:
            plt.show(block=False)

    # =======================================================
    #                      Plotting Tools
    # =======================================================
    def get_label(self, key):
        """Return formatted string for plot label
        """
        return self.config['plotting']['labels'].get(key, key)

    def _get_trans_xy(self, chk, key, x_var, y):
        """Return x, y points of transition line, for given x-axis variable

        parameters
        ----------
        chk : int
        x_var : str
        y : []
            1D array of y-values
        """
        # TODO: rename get_trans_x()
        y_max = np.max(y)
        y_min = np.min(y)

        # y_min = -2e18  # TODO: automagic this
        # y_max = 2e18

        x = self._get_trans_x(chk=chk, key=key, x_var=x_var)
        x = [x, x]
        y = [y_min, y_max]
        return x, y

    def _get_trans_x(self, chk, key, x_var):
        """Return x value corresponding to given transition
        """
        profile = self.profiles[chk]
        trans_idx = self.chk_table.loc[chk, f'{key}_i']
        return profile[x_var][trans_idx]

    def _plot_trans_line(self, x_var, y, ax, chk, trans, linewidth=1):
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
            for key in self.trans:
                x, y = self._get_trans_xy(chk=chk, key=key, x_var=x_var, y=y)
                ax.plot(x, y, ls='--', color='k', linewidth=linewidth)

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
            time = dt * chk - self.bounce_time
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

    def _setup_slider_fig(self, figsize):
        """Setup fig, ax for slider
        """
        c = self.config['plotting']  # TODO: default settings from config
        fig = plt.figure(figsize=figsize)
        profile_ax = fig.add_axes([0.1, 0.2, 0.8, 0.65])
        slider_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05])

        return fig, profile_ax, slider_ax

    def _get_slider_chk(self):
        """Return chk_max, chk_min, chk_init
        """
        chk_max = self.chk_table.index[-1]
        chk_min = self.chk_table.index[0]
        chk_init = chk_max
        return chk_max, chk_min, chk_init
