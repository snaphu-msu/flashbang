"""Main flashbang class

The Simulation object represents a single 1D FLASH model.
It can load model datafiles, manipulate/extract that data,
and plot it across various axes.

Expected model directory structure
----------------------------------
$FLASH_MODELS
│
└───<model_set>
|   |
|   └───<model>
|   │   │   <run>.dat
|   │   │   <run>.log
|   │   │   ...
|   │   │
|   │   └───output
|   │       │   <run>_hdf5_chk_0000
|   │       │   <run>_hdf5_chk_0001
|   │       │   ...


Nomenclature
-------------------
    Setup arguments
    ---------------
    model_set: name of directory containing the set of models
        If <model> is directly below $FLASH_MODELS,
        i.e. there is no <model_set> level, just use model_set=''.

    model: name of the model directory
        Typically corresponds to a particular compiled `flash4` executable.

    run: sub-model label (the actual prefix used in filenames)
        This can also be used to distinguish between multiple "runs"
        executed under the same umbrella "model".


    Data objects
    ---------------
    dat: Integrated time-series quantities found in the `<run>.dat` file.

    chk: Checkpoint data found in `chk` files.

    profile: Radial profiles extracted from chk files.

    log: Diagnostics printed during simulation, found in the `<run>.log` file.

    tracers: Time-dependent trajectories/tracers for given mass shells.
        Extracted from profiles for a chosen mass grid.
"""
import time
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib.widgets import Slider

# flashbang
from . import load_save
from . import plot_tools
from .quantities import get_density_zone
from .paths import model_path
from .tools import ensure_sequence


class Simulation:
    def __init__(self, run, model, model_set, config='default',
                 verbose=True, load_all=True,
                 reload=False, save=True, load_tracers=False):
        """Object representing a 1D flash simulation

        parameters
        ----------
        run : str
            The label that's used in chk and .dat filenames, e.g. 'run' for 'run.dat'
        model : str
            The name of the main model directory
        model_set : str
            Higher-level label of model collection
        config : str
            Base name of config file to use, e.g. 'default' for 'config/default.ini'
        load_all : bool
            Immediately load all model data (chk profiles, dat)
        load_tracers : bool
            Extract mass tracers/trajectories from profiles
        reload : bool
            Force reload from raw model files (don't load from cache/)
        save : bool
            Save extracted model data to temporary files (for faster loading)
        verbose : bool
            Print information to terminal
        """
        t0 = time.time()
        self.verbose = verbose
        self.run = run
        self.model = model
        self.model_set = model_set
        self.model_path = model_path(model, model_set=model_set)

        self.dat = None                  # time-integrated data from .dat; see load_dat()
        self.bounce_time = None          # core-bounce in simulation time (s)
        self.trans_dens = None           # transition densities (helmholtz models)
        self.mass_grid = None            # mass shells of tracers
        self.chk_table = pd.DataFrame()  # scalar chk quantities (trans_dens, time, etc.)
        self.profiles = xr.Dataset()     # radial profile data for each timestep
        self.tracers = None              # mass tracers/trajectories

        self.config = load_save.load_config(config, verbose=self.verbose)
        self.trans_dens = self.config['transitions']['dens']
        self.setup_mass_grid()
        self.load_chk_table(reload=reload, save=save)

        if load_all:
            self.load_all(reload=reload, save=save)
        if load_tracers:
            self.get_tracers(reload=reload, save=save)

        t1 = time.time()
        self.printv(f'Model load time: {t1-t0:.3f} s')

    # =======================================================
    #                      Setup/init
    # =======================================================
    def setup_mass_grid(self):
        """Generate mass grid from config definition
        """
        mass_def = self.config['tracers']['mass_grid']
        self.mass_grid = np.linspace(mass_def[0], mass_def[1], mass_def[2])

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
        self.get_transition_zones(reload=reload, save=save)

    def get_bounce_time(self):
        """Get bounce time (s) from log file
        """
        self.bounce_time = load_save.get_bounce_time(run=self.run,
                                                     model=self.model,
                                                     model_set=self.model_set,
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
        self.chk_table = load_save.get_chk_table(run=self.run,
                                                 model=self.model,
                                                 model_set=self.model_set,
                                                 reload=reload,
                                                 save=save,
                                                 verbose=self.verbose)
        self.check_chk_table(save=save)

    def load_dat(self, reload=False, save=True):
        """Load .dat file

        parameters
        ----------
        reload : bool
        save : bool
        """
        self.dat = load_save.get_dat(run=self.run,
                                     model=self.model,
                                     model_set=self.model_set,
                                     cols_dict=self.config['dat_columns'],
                                     reload=reload,
                                     save=save,
                                     verbose=self.verbose)

    def load_all_profiles(self, reload=False, save=True):
        """Load profiles for all available checkpoints

        parameters
        ----------
        reload : bool
        save : bool
        """
        config = self.config['profiles']

        self.profiles = load_save.get_multiprofile(
                                run=self.run,
                                model=self.model,
                                model_set=self.model_set,
                                chk_list=self.chk_table.index,
                                params=config['params'] + config['isotopes'],
                                derived_params=config['derived_params'],
                                reload=reload,
                                save=save,
                                verbose=self.verbose)

    def check_chk_table(self, save=True):
        """Checks that pre-saved data is up to date with any new chk files
        """
        chk_list = load_save.find_chk(run=self.run,
                                      model=self.model,
                                      model_set=self.model_set)

        if len(chk_list) != len(self.chk_table):
            self.printv('chk files missing from table, reloading')
            self.load_chk_table(reload=True, save=save)

    def save_chk_table(self):
        """Saves chk_table DataFrame to file
        """
        load_save.save_cache('chk_table',
                             data=self.chk_table,
                             run=self.run,
                             model=self.model,
                             model_set=self.model_set,
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
            idx_list = np.zeros_like(self.chk_table.index)

            for i, chk in enumerate(self.chk_table.index):
                density = self.profiles.sel(chk=chk)['dens']
                idx_list[i] = get_density_zone(density, trans_dens)

            self.chk_table[f'{key}_i'] = idx_list

    def get_tracers(self, reload=False, save=True):
        """Construct mass tracers from profile data (or load pre-extracted)

        parameters
        ----------
        reload : bool
        save : bool
        """
        if len(self.config['tracers']['params']) == 0:
            return

        self.tracers = load_save.get_tracers(run=self.run,
                                             model=self.model,
                                             model_set=self.model_set,
                                             mass_grid=self.mass_grid,
                                             params=self.config['tracers']['params'],
                                             profiles=self.profiles,
                                             reload=reload,
                                             save=save,
                                             verbose=self.verbose)

        # force reload if chks are missing
        if not np.array_equal(self.tracers.coords['chk'], self.chk_table.index):
            self.printv('Profiles missing from tracers; re-extracting')
            self.get_tracers(reload=True, save=save)

    # =======================================================
    #                      Plotting
    # =======================================================
    def plot_profiles(self, chk, y_var_list, x_var='r', y_scale=None, x_scale=None,
                      max_cols=2, sub_figsize=(6, 5), trans=False, legend=False,
                      title=True):
        """Plot one or more profile variables

        parameters
        ----------
        chk : int
            checkpoint to plot
        y_var_list : str or [str]
            variable(s) to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        legend : bool
        max_cols : bool
        sub_figsize : tuple
        trans : bool
        title : bool
        """
        chk = ensure_sequence(chk)
        y_var_list = ensure_sequence(y_var_list)
        n_var = len(y_var_list)
        fig, ax = plot_tools.setup_subplots(n_var, max_cols=max_cols, sharex=True,
                                            sub_figsize=sub_figsize, squeeze=False)

        for i, y_var in enumerate(y_var_list):
            row = int(np.floor(i / max_cols))
            col = i % max_cols
            show_legend = legend if i == 0 else False
            show_title = title if i == 0 else False

            self.plot_profile(chk=chk,
                              y_var=y_var, x_var=x_var,
                              y_scale=y_scale, x_scale=x_scale,
                              ax=ax[row, col], trans=trans,
                              legend=show_legend, title=show_title)
        return fig

    def plot_profile(self, chk, y_var, x_var='r', y_scale=None, x_scale=None,
                     ax=None, legend=False, trans=False, title=True,
                     ylims=None, xlims=None, label=None,
                     linestyle='-', marker='', title_str=None, color=None,
                     data_only=False, y_factor=1):
        """Plot given profile variable

        parameters
        ----------
        chk : int or [int]
            checkpoint(s) to plot
        y_var : str
            variable to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        y_factor : float
        ax : Axes
        legend : bool
        trans : bool
        title : bool
        ylims : [min, max]
        xlims : [min, max]
        label : str
        linestyle : str
        marker : str
        title_str : str
        color : str
        data_only : bool
            only plot data, neglecting all titles/labels/scales
        """
        chk = ensure_sequence(chk)
        fig, ax = plot_tools.setup_fig(ax=ax)

        for i in chk:
            profile = self.profiles.sel(chk=i)
            y = profile[y_var]
            ax.plot(profile[x_var], y/y_factor, ls=linestyle, marker=marker,
                    label=label, color=color)

            self._plot_trans_line(x_var, y=y, ax=ax, chk=i, trans=trans)

        if not data_only:
            self._set_ax_all(ax, x_var=x_var, y_var=y_var, xlims=xlims, ylims=ylims,
                             x_scale=x_scale, y_scale=y_scale, chk=chk[0], title=title,
                             title_str=title_str, legend=legend)

        return fig

    def plot_composition(self, chk, x_var='r', y_var_list=None, y_scale='linear',
                         x_scale=None, ax=None, legend=True, trans=True,
                         ylims=(1e-7, 1), xlims=None,
                         title=True, loc=3, data_only=False):
        """Plot isotope composition profile

        parameters
        ----------
        chk : int
        x_var : str
            variable to plot on x-axis
        y_var_list : [str]
            list of isotopes to plot (see self.config['profiles']['params'])
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        ax : Axes
        legend : bool
        trans : bool
        ylims : [min, max]
        xlims : [min, max]
        title : bool
        loc : str or int
        data_only : bool
        """
        if y_var_list is None:
            y_var_list = self.config['plotting']['isotopes']

        fig, ax = plot_tools.setup_fig(ax=ax)
        profile = self.profiles.sel(chk=chk)

        for y_var in y_var_list:
            ax.plot(profile[x_var], profile[y_var],
                    label=y_var, color={'ye': 'k'}.get(y_var),
                    linestyle={'ye': '--'}.get(y_var))

        self._plot_trans_line(x_var, y=ylims, ax=ax, chk=chk, trans=trans)

        if not data_only:
            self._set_ax_all(ax, x_var=x_var, y_var='$X$', xlims=xlims, ylims=ylims,
                             x_scale=x_scale, y_scale=y_scale,
                             chk=chk, title=title, legend=legend, loc=loc)

        return fig

    def plot_dat(self, y_var, x_scale=None, y_scale=None, ax=None,
                 linestyle='-', marker='', label=None, legend=False,
                 zero_time=True, title_str=None, xlims=None, ylims=None,
                 color=None, x_factor=1, y_factor=1, data_only=False):
        """Plot quantity from dat file

        parameters
        ----------
        y_var : str
        x_scale : 'log' or 'linear'
        y_scale : 'log' or 'linear'
        ax : Axes
        linestyle : str
        marker : str
        label : str
        legend : bool
        zero_time : bool
        title_str : str
        xlims : [min, max]
        ylims : [min, max]
        color : str
        x_factor : float
        y_factor : float
        data_only : bool
        """
        t_offset = 0
        if zero_time:
            t_offset = self.bounce_time

        fig, ax = plot_tools.setup_fig(ax=ax)

        ax.plot((self.dat['time'] - t_offset) / x_factor,
                self.dat[y_var] / y_factor,
                linestyle=linestyle, marker=marker,
                color=color, label=label)

        if not data_only:
            self._set_ax_all(ax, x_var='time', y_var=y_var, xlims=xlims, ylims=ylims,
                             x_scale=x_scale, y_scale=y_scale,
                             title=True, title_str=title_str, legend=legend)

        return fig

    def plot_tracers(self, y_var, x_scale=None, y_scale=None, ax=None,
                     xlims=None, ylims=None, linestyle='-', marker='',
                     legend=False, data_only=False):
        """Plot quantity from dat file

        parameters
        ----------
        y_var : str
        x_scale : 'log' or 'linear'
        y_scale : 'log' or 'linear'
        xlims : [min, max]
        ylims : [min, max]
        ax : Axes
        linestyle : str
        marker : str
        legend : bool
        data_only : bool
        """
        fig, ax = plot_tools.setup_fig(ax=ax)

        for mass in self.tracers['mass']:
            ax.plot(self.tracers['chk'], self.tracers.sel(mass=mass)[y_var],
                    linestyle=linestyle, marker=marker, label=f'{mass.values:.3f}')

        if not data_only:
            self._set_ax_all(ax, x_var='chk', y_var=y_var, xlims=xlims, ylims=ylims,
                             x_scale=x_scale, y_scale=y_scale, title=False, legend=legend)

        return fig

    # =======================================================
    #                      Sliders
    # =======================================================
    def plot_profile_slider(self, y_var, x_var='r', y_scale=None, x_scale=None,
                            xlims=None, ylims=None, trans=False,
                            title=True,  legend=False, linestyle='-',
                            marker='', y_factor=1):
        """Plot interactive slider of profile for given variable

        parameters
        ----------
        y_var : str
        x_var : str
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        trans : bool
            plot helmholtz transitions
        title : bool
        xlims : [min, max]
        ylims : [min, max]
        legend : bool
        linestyle : str
        marker : str
        y_factor : float
        """
        def update_slider(chk):
            idx = int(chk)
            profile = self.profiles.sel(chk=idx)
            y_profile = profile[y_var] / y_factor

            self._update_ax_line(x=profile[x_var], y=y_profile, line=lines[y_var])
            self._set_ax_title(profile_ax, chk=idx, title=title)

            if trans:
                for trans_key in self.trans_dens:
                    x, y = self._get_trans_xy(chk=idx, key=trans_key,
                                              x_var=x_var, y=y_profile)
                    self._update_ax_line(x=x, y=y, line=lines[trans_key])

            fig.canvas.draw_idle()

        fig, profile_ax, slider_ax = plot_tools.setup_slider_fig()
        chk_max, chk_min = self._get_slider_chk()

        slider = Slider(slider_ax, 'chk', chk_min, chk_max, valinit=chk_max, valstep=1)

        self.plot_profile(chk=chk_max,
                          y_var=y_var, x_var=x_var,
                          y_scale=y_scale, x_scale=x_scale,
                          ylims=ylims, xlims=xlims,
                          ax=profile_ax, legend=legend,
                          trans=trans, title=title,
                          linestyle=linestyle,
                          marker=marker, y_factor=y_factor)

        lines = self._get_ax_lines(ax=profile_ax, y_vars=[y_var], trans=trans)
        slider.on_changed(update_slider)

        return fig, slider

    def plot_composition_slider(self, y_var_list=None, x_var='r', y_scale='linear',
                                x_scale=None, trans=True, title=True,
                                xlims=None, ylims=(1e-7, 1), legend=True,
                                loc='lower left'):
        """Plot interactive slider of isotope composition

        parameters
        ----------
        y_var_list : [str]
        x_var : str
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        trans : bool
            plot helmholtz transitions
        title : bool
        xlims : [min, max]
        ylims : [min, max]
        legend : bool
        loc : str
        """
        def update_slider(chk):
            idx = int(chk)
            profile = self.profiles.sel(chk=idx)

            if trans:
                for i, key in enumerate(self.trans_dens):
                    x, y = self._get_trans_xy(chk=idx, key=key, x_var=x_var, y=ylims)
                    profile_ax.lines[-i-1].set_xdata(x)
                    profile_ax.lines[-i-1].set_ydata(y)

            for i, key in enumerate(y_var_list):
                y_profile = profile[key]
                profile_ax.lines[i].set_xdata(profile[x_var])
                profile_ax.lines[i].set_ydata(y_profile)

            self._set_ax_title(profile_ax, chk=idx, title=title)
            fig.canvas.draw_idle()

        fig, profile_ax, slider_ax = plot_tools.setup_slider_fig()
        chk_max, chk_min = self._get_slider_chk()

        if y_var_list is None:
            y_var_list = self.config['plotting']['isotopes']

        slider = Slider(slider_ax, 'chk', chk_min, chk_max, valinit=chk_max, valstep=1)

        self.plot_composition(chk_max, x_var=x_var, y_scale=y_scale, x_scale=x_scale,
                              y_var_list=y_var_list, ax=profile_ax, legend=legend,
                              ylims=ylims, xlims=xlims, trans=trans, title=title,
                              loc=loc)

        slider.on_changed(update_slider)

        return fig, slider

    # =======================================================
    #                      Plotting Tools
    # =======================================================
    def _get_trans_xy(self, chk, key, x_var, y):
        """Return x, y points of transition line, for given x-axis variable

        parameters
        ----------
        chk : int
        key : str
        x_var : str
        y : 1D array
        """
        y_max = np.max(y)
        y_min = np.min(y)

        x = self._get_trans_x(chk=chk, key=key, x_var=x_var)
        x = np.array([x, x])
        y = np.array([y_min, y_max])
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

    def _set_ax_all(self, ax, x_var, y_var, x_scale, y_scale,
                    xlims, ylims, title, legend, loc=None,
                    chk=None, title_str=None):
        """Set all axis properties

        parameters
        ----------
        ax : Axes
        x_var : str
        y_var : str
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        chk : int
        title : bool
        title_str : str
        xlims : [min, max]
        ylims : [min, max]
        legend : bool
        loc : int or str
        """
        self._set_ax_title(ax, chk=chk, title=title, title_str=title_str)
        self._set_ax_lims(ax, xlims=xlims, ylims=ylims)
        self._set_ax_labels(ax, x_var=x_var, y_var=y_var)
        self._set_ax_legend(ax, legend=legend, loc=loc)
        self._set_ax_scales(ax, x_var=x_var, y_var=y_var,
                            x_scale=x_scale, y_scale=y_scale)

    def _set_ax_scales(self, ax, y_var, x_var, y_scale, x_scale):
        """Set axis scales (linear, log)

        parameters
        ----------
        ax : Axes
        y_var : str
        x_var : str
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        """
        def get_scale(var):
            if var in self.config['plotting']['ax_scales']['log']:
                return 'log'
            else:
                return 'linear'

        if x_scale is None:
            x_scale = get_scale(x_var)
        if y_scale is None:
            y_scale = get_scale(y_var)

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

    def _set_ax_title(self, ax, title, chk=None, title_str=None):
        """Set axis title

        parameters
        ----------
        ax : Axes
        chk : int
        title : bool
        title_str : str
        """
        if title:
            if (title_str is None) and (chk is not None):
                # timestep = self.chk_table.loc[chk, 'time'] - self.bounce_time
                dt = self.config['plotting']['scales']['chk_dt']
                timestep = dt * chk - self.bounce_time
                title_str = f't = {timestep:.3f} s'

            ax.set_title(title_str)

    def _set_ax_lims(self, ax, xlims, ylims):
        """Set x and y axis limits

        parameters
        ----------
        ax : Axes
        xlims : [min, max]
        ylims : [min, max]
        """
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
        def get_label(key):
            return self.config['plotting']['labels'].get(key, key)

        ax.set_xlabel(get_label(x_var))
        ax.set_ylabel(get_label(y_var))

    def _set_ax_legend(self, ax, legend, loc=None):
        """Set axis labels

        parameters
        ----------
        ax : Axes
        legend : bool
        """
        if legend:
            ax.legend(loc=loc)

    def _get_slider_chk(self):
        """Return chk_max, chk_min
        """
        chk_max = self.chk_table.index[-1]
        chk_min = self.chk_table.index[0]
        return chk_max, chk_min

    def _get_ax_lines(self, ax, y_vars, trans):
        """Return dict of axis line indexes

        Parameters
        ----------
        ax : Axis
        y_vars : [str]
        trans : bool
        """
        lines = {}
        n_vars = len(y_vars)

        for i, y_var in enumerate(y_vars):
            lines[y_var] = ax.lines[i]

        if trans:
            for i, trans_key in enumerate(self.trans_dens):
                lines[trans_key] = ax.lines[n_vars+i]

        return lines

    def _update_ax_line(self, x, y, line):
        """Update x,y line values

        Parameters
        ----------
        x : array
        y : array
        line : Axis.line
        """
        line.set_xdata(x)
        line.set_ydata(y)

    # =======================================================
    #                   Convenience
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
