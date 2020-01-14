"""Main flashbang class for the Simulation object.

A Simulation instance represents a single 1D FLASH model.
It can load model data files, manipulate/extract that data,
and plot it across various "dimensions".

General terminology
-------------------
    Setup arguments
    ---------------
    model: Name of the FLASH model (i.e. the directory name).
        Typically corresponds to a particular compiled `flash4` executable.

    run: sub-model label (i.e. the prefix used in filenames,
        e.g. use 'run2' for 'run2.dat').
        Multiple simulations may have been executed under the
        same umbrella "model". Use this to distinguish between them.

    Data structures
    ---------------
    dat: Integrated time-series quantities found in the `[run].dat` file.

    chk: Checkpoint data found in `chk` files.

    profile: Radial profiles as extracted from chk files.
        Each profile corresponds to a chk file.

    log: Data printed to terminal during model, stored in the `[run].log` file.

    tracers: Trajectories/tracers for given mass shells.
        Extracted using profile mass coordinates, for a chosen mass grid.
"""
import os
import time
import numpy as np
import xarray as xr
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
#   - change mass units to m_sun
#   - chk_table, add columns: (add to multiprofile metadata?)
#       - time
#       - n_step
#       - n_zones
#       - rsh_avg, other dat params
#   - plotting:
#       - find max/min y-values over all chk --> set slider ylim
#       - plot tracers


# noinspection PyTypeChecker
class Simulation:
    def __init__(self, model, run='run', config='default',
                 output_dir='output', verbose=True, load_all=True,
                 reload=False, save=True):
        """Object representing a 1D flash simulation

        parameters
        ----------
        model : str
            The name of the main model directory

        run : str
            The label that's used in chk and .dat filenames, e.g. 'run' for 'run.dat'

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
        """
        # TODO: comment attrs
        t0 = time.time()
        self.verbose = verbose
        self.model = model
        self.run = run

        self.model_path = paths.model_path(model=model)
        self.output_path = os.path.join(self.model_path, output_dir)

        self.config = None
        self.dat = None
        self.bounce_time = None
        self.trans_dens = None
        self.n_chk = None
        self.mass_grid = None
        self.chk_table = pd.DataFrame()
        self.profiles = xr.Dataset()
        self.tracers = None

        self.load_config(config=config)
        self.load_chk_table(reload=reload, save=save)

        if load_all:
            self.load_all(reload=reload, save=save)

        t1 = time.time()
        self.printv(f'Model load time: {t1-t0:.3f} s')

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
            override self.verbose setting
        **kwargs
            args for print()
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print(string, **kwargs)

    def load_config(self, config='default'):
        """Load config parameters from file

        parameters
        ----------
        config : str
        """
        self.config = load_save.load_config(name=config, verbose=self.verbose)
        self.trans_dens = self.config['transitions']['dens']
        self.setup_mass_grid()

    def setup_mass_grid(self):
        """Generate mass grid from config definition
        """
        mass_def = self.config['tracers']['mass_grid']
        self.mass_grid = analysis.get_mass_grid(mass_low=mass_def[0],
                                                mass_high=mass_def[1],
                                                n_points=mass_def[2])

    def load_all(self, reload=False, save=True):
        """Load all model data

        parameters
        ----------
        reload : bool
        save : bool
        """
        self.get_bounce_time()
        self.load_dat(reload=reload, save=save)
        self.load_all_profiles(reload=reload, save=save)
        self.get_tracers(reload=reload, save=save)
        self.get_transition_zones(reload=reload, save=save)

    def get_bounce_time(self):
        """Get bounce time (s) from log file
        """
        self.bounce_time = load_save.get_bounce_time(self.model, run=self.run,
                                                     verbose=self.verbose)

    # =======================================================
    #                   Loading Data
    # =======================================================
    def load_chk_table(self, reload=False, save=True):
        """Load DataFrame of chk scalars

        parameters
        ----------
        reload : bool
        save : bool
        """
        self.chk_table = load_save.get_chk_table(model=self.model, run=self.run,
                                                 reload=reload, save=save,
                                                 verbose=self.verbose)
        self.check_chk_list(save=save)

    def load_dat(self, reload=False, save=True):
        """Load .dat file

        parameters
        ----------
        reload : bool
        save : bool
        """
        self.dat = load_save.get_dat(
                        model=self.model, run=self.run,
                        cols_dict=self.config['dat_columns'], reload=reload,
                        save=save, verbose=self.verbose)

    def load_all_profiles(self, reload=False, save=True):
        """Load profiles for all available checkpoints

        parameters
        ----------
        reload : bool
        save : bool
        """
        config = self.config['profiles']

        self.profiles = load_save.get_multiprofile(
                                model=self.model, run=self.run,
                                chk_list=self.chk_table.index,
                                params=config['params'] + config['isotopes'],
                                derived_params=config['derived_params'],
                                reload=reload, save=save, verbose=self.verbose)

    def check_chk_table(self, save=True):
        """Checks that pre-saved data is up to date with any new chk files
        """
        # TODO: check other consistency (tracers, etc.)
        chk_list = load_save.find_chk(model=self.model, run=self.run)

        if len(chk_list) != len(self.chk_table):
            self.printv('chk files missing from table, reloading')
            self.load_chk_table(reload=True, save=save)

    def save_chk_table(self):
        """Saves chk_table DataFrame to file
        """
        load_save.save_chk_table_cache(self.chk_table, model=self.model, run=self.run,
                                       verbose=self.verbose)

    # =======================================================
    #                 Analysis & Postprocessing
    # =======================================================
    def get_transition_zones(self, reload=False, save=True):
        """Handles obtaining density transition zones (if specified)

        parameters
        ----------
        reload : bool
        save : bool
        """
        if self.trans_dens is not None:
            trans_missing = False

            # check if already in chk_table
            for key in self.trans_dens:
                if f'{key}_i' not in self.chk_table.columns:
                    trans_missing = True

            if trans_missing or reload:
                self.find_trans_idxs()

                if save:
                    self.save_chk_table()

    def find_trans_idxs(self):
        """Find indexes of zones closest to specified transition densities,
        for each profile timestep
        """
        self.printv('Finding transition zones')

        for key, trans_dens in self.trans_dens.items():
            idx_list = np.zeros(self.n_chk, dtype=int)

            for i, chk in enumerate(self.chk_table.index):
                profile = self.profiles.sel(chk=chk)
                dens_reverse = np.flip(profile['dens'])  # need increasing density
                max_idx = len(dens_reverse) - 1

                trans_idx = tools.find_nearest_idx(dens_reverse, trans_dens)
                idx_list[i] = max_idx - trans_idx  # flip back

            self.chk_table[f'{key}_i'] = idx_list

    def get_tracers(self, reload=False, save=True):
        """Construct mass tracers from profile data (or load pre-extracted)

        parameters
        ----------
        reload : bool
        save : bool
        """
        # TODO:
        #   - include chk timesteps
        self.tracers = load_save.get_tracers(model=self.model, run=self.run,
                                             mass_grid=self.mass_grid,
                                             params=self.config['tracers']['params'],
                                             profiles=self.profiles,
                                             reload=reload, save=save,
                                             verbose=self.verbose)

        # force reload if chks are missing
        if not np.array_equal(self.tracers.coords['chk'], self.chk_table.index):
            self.printv('Profiles missing from tracers; re-extracting')
            self.get_tracers(reload=True, save=save)

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
        y_var_list : str or [str]
            variable(s) to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
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
        chk : int or [int]
            checkpoint(s) to plot
        y_var : str
            variable to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
        ax : Axes
        legend : bool
        trans : bool
        title : bool
        ylims : [min, max]
        xlims : [min, max]
        figsize : [width, height]
        label : str
        linestyle : str
        marker : str
        """
        chk = tools.ensure_sequence(chk)

        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        self._set_ax_title(ax, chk=chk[0], title=title)
        self._set_ax_scales(ax, y_var, x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_lims(ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(ax, x_var=x_var, y_var=y_var)

        for i in chk:
            profile = self.profiles.sel(chk=i)
            y = profile[y_var]

            ax.plot(profile[x_var], y, ls=linestyle, marker=marker, label=label)
            self._plot_trans_line(x_var, y=y, ax=ax, chk=i, trans=trans)

        if legend:
            ax.legend()

        return fig

    def plot_composition(self, chk, x_var='r', y_var_list=None, y_scale='linear',
                         x_scale=None, ax=None, legend=True, trans=True, show_ye=True,
                         ylims=(1e-7, 1), xlims=(1e5, 1.5e9), figsize=(8, 6),
                         title=True, legend_loc='lower left'):
        """Plot isotope composition profile

        parameters
        ----------
        chk : int
        x_var : str
            variable to plot on x-axis
        y_var_list : [str]
            list of isotopes to plot (see self.config['profiles']['params'])
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
        ax : Axes
        legend : bool
        trans : bool
        show_ye : bool
        ylims : [min, max]
        xlims : [min, max]
        figsize : [width, height]
        title : bool
        legend_loc : str or int
        """
        if y_var_list is None:
            y_var_list = self.config['plotting']['isotopes']

        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        self._set_ax_scales(ax, y_var_list[0], x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_title(ax, chk=chk, title=title)
        self._set_ax_lims(ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(ax, x_var=x_var, y_var='$X$')

        profile = self.profiles.sel(chk=chk)
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
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
        trans : bool
            plot helmholtz transitions
        figsize : [width, height]
        title : bool
        xlims : [min, max]
        ylims : [min, max]
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
            profile = self.profiles.sel(chk=idx)
            y_profile = profile[y_var]

            profile_ax.lines[0].set_ydata(y_profile)
            profile_ax.lines[0].set_xdata(profile[x_var])
            self._set_ax_title(profile_ax, chk=idx, title=title)

            if trans:
                for i, key in enumerate(self.trans_dens):
                    x, y = self._get_trans_xy(chk=idx, key=key, x_var=x_var, y=y_profile)
                    profile_ax.lines[i+1].set_xdata(x)
                    profile_ax.lines[i+1].set_ydata(y)

            fig.canvas.draw_idle()

        slider.on_changed(update)
        return fig, slider

    def plot_slider_composition(self, y_var_list=None, x_var='r', y_scale='linear',
                                x_scale=None, trans=True, figsize=(8, 6), title=True,
                                xlims=(1e5, 1.5e9), ylims=(1e-7, 1), legend=True,
                                show_ye=True, legend_loc='lower left'):
        """Plot interactive slider of isotope composition

        parameters
        ----------
        y_var_list : [str]
        x_var : str
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
        trans : bool
            plot helmholtz transitions
        figsize : [width, height]
        title : bool
        xlims : [min, max]
        ylims : [min, max]
        legend : bool
        show_ye : bool
        legend_loc : str
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
                              show_ye=show_ye, legend_loc=legend_loc)

        def update(chk):
            idx = int(chk)
            profile = self.profiles.sel(chk=idx)

            if trans:
                # TODO: nicer way to do this
                for i, key in enumerate(self.trans_dens):
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
        y_scale : {'log', 'linear'}
        figsize : [width, height]
        display : bool
        ax : Axes
        linestyle : str
        marker : str
        """
        # TODO: subtract bounce_time
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

        parameters
        ----------
        key : str
            parameter key, e.g. 'r', 'temp', 'dens'
        """
        return self.config['plotting']['labels'].get(key, key)

    def _get_trans_xy(self, chk, key, x_var, y):
        """Return x, y points of transition line, for given x-axis variable

        parameters
        ----------
        chk : int
        key : str
        x_var : str
        y : 1D array
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

        parameters
        ----------
        chk : int
        key : str
        x_var : str
        """
        profile = self.profiles.sel(chk=chk)
        trans_idx = self.chk_table.loc[chk, f'{key}_i']
        return profile[x_var][trans_idx]

    def _plot_trans_line(self, x_var, y, ax, chk, trans, linewidth=1):
        """Add transition line to axis

        parameters
        ----------
        x_var : str
        y : []
            1D array of y-values
        ax : Axes
        chk : int
        trans : bool
        """
        if trans:
            for key in self.trans_dens:
                x, y = self._get_trans_xy(chk=chk, key=key, x_var=x_var, y=y)
                ax.plot(x, y, ls='--', color='k', linewidth=linewidth)

    def _set_ax_scales(self, ax, y_var, x_var, y_scale, x_scale):
        """Set axis scales (linear, log)

        parameters
        ----------
        ax : Axes
        y_var : str
        x_var : str
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
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
        ax : Axes
        chk : int
        title : bool
        """
        # TODO: account for different zero points/starting times
        if title:
            dt = self.config['plotting']['scales']['chk_dt']
            timestep = dt * chk - self.bounce_time
            ax.set_title(f't = {timestep:.3f} s')

    def _set_ax_lims(self, ax, xlims, ylims):
        """Set x and y axis limits

        parameters
        ----------
        ax : Axes
        xlims : [min, max]
        ylims : [min, max]
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
        ax : Axes
        x_var : str
        y_var : str
        """
        ax.set_xlabel(self.get_label(x_var))
        ax.set_ylabel(self.get_label(y_var))

    def _setup_fig_ax(self, ax, figsize):
        """Setup fig, ax, checking if ax already provided

        parameters
        ----------
        ax : Axes
        figsize : [width, height]
        """
        c = self.config['plotting']  # TODO: default settings from config
        fig = None

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        return fig, ax

    def _setup_slider_fig(self, figsize):
        """Setup fig, ax for slider

        parameters
        ----------
        figsize : [width, height]
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
