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

# flashbang
from . import load_save
from .plotting import plot_tools
from .plotting.plotter import Plotter
from .plotting.slider import FlashSlider
from .quantities import get_density_zone
from .paths import model_path
from .tools import ensure_sequence
from .config import Config


class Simulation:
    def __init__(self,
                 run,
                 model,
                 model_set,
                 config=None,
                 verbose=True,
                 load_all=True,
                 reload=False,
                 save=True,
                 load_tracers=False):
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
            Name of config file to use, e.g. 'stir' for 'config/stir.ini'
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
        self.bounce = {}                 # bounce properties
        self.trans_dens = None           # transition densities (helmholtz models)
        self.mass_grid = None            # mass shells of tracers
        self.chk_table = pd.DataFrame()  # scalar chk quantities (trans_dens, time, etc.)
        self.profiles = xr.Dataset()     # radial profile data for each timestep
        self.tracers = None              # mass tracers/trajectories
        self.timesteps = None            # table of chk timesteps

        self.config = Config(name=config, verbose=self.verbose)
        self.trans_dens = self.config.trans('dens')
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
        mass_grid = self.config.tracers('mass_grid')
        self.mass_grid = np.linspace(mass_grid[0], mass_grid[1], mass_grid[2])

    def load_all(self, reload=False, save=True):
        """Load all model data

        parameters
        ----------
        reload : bool
        save : bool
        """
        self.load_dat(reload=reload, save=save)
        self.load_all_profiles(reload=reload, save=save)
        self.get_transition_zones(reload=reload, save=save)
        self.load_timesteps(reload=reload, save=save)

        self.get_bounce_time()
        self.get_bounce_chk()
        self.get_bounce_core_mass()

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
                                     config=self.config,
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
        self.profiles = load_save.get_multiprofile(
                                run=self.run,
                                model=self.model,
                                model_set=self.model_set,
                                chk_list=self.chk_table.index,
                                config=self.config,
                                reload=reload,
                                save=save,
                                verbose=self.verbose)

    def get_bounce_time(self):
        """Get bounce time (s) from log file
        """
        self.bounce['time'] = load_save.get_bounce_time(run=self.run,
                                                        model=self.model,
                                                        model_set=self.model_set,
                                                        verbose=self.verbose)

    def save_chk_table(self):
        """Saves chk_table DataFrame to file
        """
        load_save.save_cache('chk_table',
                             data=self.chk_table,
                             run=self.run,
                             model=self.model,
                             model_set=self.model_set,
                             verbose=self.verbose)

    def load_timesteps(self, reload=False, save=True):
        """Load table of chk timesteps
        """
        self.timesteps = load_save.get_timesteps(run=self.run,
                                                 model=self.model,
                                                 model_set=self.model_set,
                                                 reload=reload,
                                                 save=save,
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
        params = self.config.tracers('params')
        if len(params) == 0:
            return

        self.tracers = load_save.get_tracers(run=self.run,
                                             model=self.model,
                                             model_set=self.model_set,
                                             mass_grid=self.mass_grid,
                                             params=params,
                                             profiles=self.profiles,
                                             reload=reload,
                                             save=save,
                                             verbose=self.verbose)

        # force reload if chks are missing
        if not np.array_equal(self.tracers.coords['chk'], self.chk_table.index):
            self.printv('Profiles missing from tracers; re-extracting')
            self.get_tracers(reload=True, save=save)

    def get_bounce_chk(self):
        """Find chk closest to bounce
        """
        dt = (self.timesteps['time'] - self.bounce['time']).abs()
        chk = dt.idxmin()
        chk_dt = self.timesteps.loc[chk, 'time'] - self.bounce['time']

        self.bounce['chk'] = chk
        self.bounce['chk_dt'] = chk_dt

    def get_bounce_core_mass(self, max_mass=1, entr=3):
        """Get inner core mass at bounce

        Parameters
        ----------
        max_mass : float
            maximum enclosed mass to look within
        entr : float
            entropy threshold defining core mass
        """
        bounce_profile = self.profiles.sel(chk=self.bounce['chk'])

        bounce_slice = bounce_profile.where(bounce_profile.mass < max_mass, drop=True)
        mass = bounce_slice.where(bounce_slice.entr > entr, drop=True).mass

        if len(mass) is not 1:
            self.printv('Unique bounce core mass not found! '
                        'Bounce chk may not be at bounce')
            return

        self.bounce['core_mass'] = float(mass)

    # =======================================================
    #                      Plotting
    # =======================================================
    def plot_profiles(self, chk, y_vars,
                      x_var='r',
                      x_scale=None, y_scale=None,
                      x_factor=None, y_factor=None,
                      x_label=None, y_label=None,
                      legend=False, legend_loc=None,
                      title=True, title_str=None,
                      max_cols=2,
                      sub_figsize=(6, 5),
                      trans=None,
                      ):
        """Plot one or more profile variables

        parameters
        ----------
        chk : int
            checkpoint to plot
        y_vars : str or [str]
            variable(s) to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        x_factor : float
        y_factor : float
        x_label : str
        y_label : str
        legend : bool
        legend_loc : str or int
        title : bool
        title_str : str
        max_cols : bool
        sub_figsize : tuple
        trans : bool
        """
        chk = ensure_sequence(chk)
        y_vars = ensure_sequence(y_vars)
        n_var = len(y_vars)
        fig, ax = plot_tools.setup_subplots(n_var,
                                            max_cols=max_cols,
                                            sharex=True,
                                            sub_figsize=sub_figsize,
                                            squeeze=False)

        for i, y_var in enumerate(y_vars):
            row = int(np.floor(i / max_cols))
            col = i % max_cols
            show_legend = legend if i == 0 else False
            show_title = title if i == 0 else False

            self.plot_profile(chk=chk,
                              y_var=y_var,
                              x_var=x_var,
                              y_scale=y_scale,
                              x_scale=x_scale,
                              x_factor=x_factor,
                              y_factor=y_factor,
                              x_label=x_label,
                              y_label=y_label,
                              ax=ax[row, col],
                              trans=trans,
                              legend=show_legend,
                              legend_loc=legend_loc,
                              title=show_title,
                              title_str=title_str)
        return fig

    def plot_profile(self, chk, y_var,
                     x_var='r',
                     x_scale=None, y_scale=None,
                     x_lims=None, y_lims=None,
                     x_factor=None, y_factor=None,
                     x_label=None, y_label=None,
                     legend=False, legend_loc=None,
                     title=True, title_str=None,
                     linestyle=None, marker=None,
                     ax=None,
                     trans=None,
                     label=None,
                     color=None,
                     data_only=False):
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
        x_factor : float
        y_factor : float
        x_label : str
        y_label : str
        legend : bool
        legend_loc : str or int
        title : bool
        title_str : str
        y_lims : [min, max]
        x_lims : [min, max]
        linestyle : str
        marker : str
        ax : Axes
        trans : bool
        label : str
        color : str
        data_only : bool
            only plot data, neglecting all titles/labels/scales
        """
        chk = ensure_sequence(chk)
        title_str = self._get_title(chk=chk[0], title_str=title_str)

        self._check_trans(trans=trans)
        if trans is None:
            trans = self.config.trans('plot')

        plot = Plotter(ax=ax,
                       config=self.config,
                       x_var=x_var,
                       y_var=y_var,
                       x_lims=x_lims,
                       y_lims=y_lims,
                       x_scale=x_scale,
                       y_scale=y_scale,
                       x_label=x_label,
                       y_label=y_label,
                       x_factor=x_factor,
                       y_factor=y_factor,
                       linestyle=linestyle,
                       marker=marker,
                       title=title,
                       title_str=title_str,
                       legend=legend,
                       legend_loc=legend_loc,
                       verbose=self.verbose)

        for i in chk:
            profile = self.profiles.sel(chk=i)
            x = profile[x_var]
            y = profile[y_var]

            plot.plot(x, y, label=label, color=color)

            if trans:
                self._plot_trans_lines(x=x, y=y, plot=plot, chk=i)

        if not data_only:
            plot.set_all()

        return plot

    def plot_composition(self, chk,
                         x_var='r', y_vars=None,
                         x_scale=None, y_scale=None,
                         x_lims=None, y_lims=None,
                         x_factor=None, y_factor=None,
                         x_label=None, y_label=None,
                         legend=True, legend_loc=None,
                         title=True, title_str=None,
                         ax=None,
                         trans=True,
                         data_only=False):
        """Plot isotope composition profile

        parameters
        ----------
        chk : int
        x_var : str
            variable to plot on x-axis
        y_vars : [str] or 'all'
            list of isotopes to plot (see Config.profiles.params)
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        x_factor : float
        y_factor : float
        x_label : str
        y_label : str
        legend : bool
        legend_loc : str or int
        title : bool
        title_str : str
        ax : Axes
        trans : bool
        y_lims : [min, max]
        x_lims : [min, max]
        data_only : bool
        """
        self._check_composition()
        self._check_trans(trans=trans)

        if y_vars is None:
            y_vars = self.config.plotting('isotopes')
        elif y_vars is 'all':
            y_vars = [x.strip() for x in self.config.profiles('isotopes')]
        if y_lims is None:
            y_lims = self.config.ax_lims('X')
        if trans is None:
            trans = self.config.trans('plot')

        title_str = self._get_title(chk=chk, title_str=title_str)

        plot = Plotter(ax=ax,
                       config=self.config,
                       x_var=x_var,
                       y_var='X',
                       x_lims=x_lims,
                       y_lims=y_lims,
                       x_scale=x_scale,
                       y_scale=y_scale,
                       x_label=x_label,
                       y_label=y_label,
                       x_factor=x_factor,
                       y_factor=y_factor,
                       title=title,
                       title_str=title_str,
                       legend=legend,
                       legend_loc=legend_loc,
                       verbose=self.verbose)

        profile = self.profiles.sel(chk=chk)
        x = profile[x_var]

        for y_var in y_vars:
            y = profile[y_var]
            color = {'ye': 'k'}.get(y_var)
            linestyle = {'ye': '--'}.get(y_var)

            plot.plot(x, y, label=y_var, color=color, linestyle=linestyle)

        if trans:
            self._plot_trans_lines(x=x, y=y_lims, plot=plot, chk=chk)

        if not data_only:
            plot.set_all()

        return plot

    def plot_dat(self, y_var,
                 x_scale=None, y_scale=None,
                 x_lims=None, y_lims=None,
                 x_factor=None, y_factor=None,
                 x_label=None, y_label=None,
                 legend=False, legend_loc=None,
                 title=True, title_str=None,
                 linestyle=None, marker=None,
                 ax=None,
                 label=None,
                 zero_time=True,
                 color=None,
                 data_only=False):
        """Plot quantity from dat file

        parameters
        ----------
        y_var : str
        x_scale : 'log' or 'linear'
        y_scale : 'log' or 'linear'
        x_lims : [min, max]
        y_lims : [min, max]
        x_factor : float
        y_factor : float
        x_label : str
        y_label : str
        legend : bool
        legend_loc : str or int
        title : bool
        title_str : str
        linestyle : str
        marker : str
        ax : Axes
        label : str
        zero_time : bool
        color : str
        data_only : bool
        """
        plot = Plotter(ax=ax,
                       config=self.config,
                       x_var='time',
                       y_var=y_var,
                       x_lims=x_lims,
                       y_lims=y_lims,
                       x_scale=x_scale,
                       y_scale=y_scale,
                       x_label=x_label,
                       y_label=y_label,
                       x_factor=x_factor,
                       y_factor=y_factor,
                       linestyle=linestyle,
                       marker=marker,
                       title=title,
                       title_str=title_str,
                       legend=legend,
                       legend_loc=legend_loc,
                       verbose=self.verbose)

        t_offset = 0
        if zero_time:
            t_offset = self.bounce['time']

        x = self.dat['time'] - t_offset
        y = self.dat[y_var]

        plot.plot(x, y, color=color, label=label)

        if not data_only:
            plot.set_all()

        return plot

    def plot_tracers(self, y_var,
                     x_scale=None, y_scale=None,
                     x_lims=None, y_lims=None,
                     x_label=None, y_label=None,
                     legend=False, legend_loc=None,
                     title=False, title_str=None,
                     ax=None,
                     linestyle='-',
                     marker='',
                     data_only=False):
        """Plot quantity from dat file

        parameters
        ----------
        y_var : str
        x_scale : 'log' or 'linear'
        y_scale : 'log' or 'linear'
        x_lims : [min, max]
        y_lims : [min, max]
        x_label : str
        y_label : str
        legend : bool
        legend_loc : str or int
        title : bool
        title_str : str
        ax : Axes
        linestyle : str
        marker : str
        data_only : bool
        """
        plot = Plotter(ax=ax,
                       config=self.config,
                       x_var='chk',
                       y_var=y_var,
                       x_lims=x_lims,
                       y_lims=y_lims,
                       x_scale=x_scale,
                       y_scale=y_scale,
                       x_label=x_label,
                       y_label=y_label,
                       title=title,
                       title_str=title_str,
                       legend=legend,
                       legend_loc=legend_loc,
                       linestyle=linestyle,
                       marker=marker,
                       verbose=self.verbose)

        for mass in self.tracers['mass']:
            x = self.tracers['chk']
            y = self.tracers.sel(mass=mass)[y_var]
            label = f'{mass.values:.3f}'

            plot.plot(x, y, label=label)

        if not data_only:
            plot.set_all()

        return plot

    # =======================================================
    #                      Sliders
    # =======================================================
    def plot_profile_slider(self, y_var,
                            x_var='r',
                            x_scale=None, y_scale=None,
                            x_lims=None, y_lims=None,
                            x_factor=None, y_factor=None,
                            x_label=None, y_label=None,
                            legend=False, legend_loc=None,
                            title=True,
                            trans=None,
                            linestyle=None,
                            marker=None):
        """Plot interactive slider of profile for given variable

        parameters
        ----------
        y_var : str
        x_var : str
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        x_lims : [min, max]
        y_lims : [min, max]
        x_factor : float
        y_factor : float
        x_label : str
        y_label : str
        trans : bool
        title : bool
        legend : bool
        legend_loc : str or int
        linestyle : str
        marker : str
        """
        def update_slider(chk):
            chk = int(chk)

            profile = self.profiles.sel(chk=chk)
            x = profile[x_var]
            y = profile[y_var]

            slider.update_ax_line(x=x, y=y, y_var=y_var)
            slider.update_trans_lines(chk=chk, x=x, y=y)

            title_str = self._get_title(chk=chk)
            plot.set_title(title_str=title_str)

            slider.fig.canvas.draw_idle()

        # ----------------
        self._check_trans(trans=trans)
        x_factor, y_factor = self._check_factors(x_var=x_var,
                                                 y_var=y_var,
                                                 x_factor=x_factor,
                                                 y_factor=y_factor)
        if trans is None:
            trans = self.config.trans('plot')

        slider = FlashSlider(y_vars=[y_var],
                             chk_table=self.chk_table,
                             trans=trans,
                             trans_dens=self.trans_dens,
                             x_factor=x_factor,
                             y_factor=y_factor)

        plot = self.plot_profile(chk=self.chk_table.index[-1],
                                 y_var=y_var,
                                 x_var=x_var,
                                 y_scale=y_scale,
                                 x_scale=x_scale,
                                 x_factor=x_factor,
                                 y_factor=y_factor,
                                 y_lims=y_lims,
                                 x_lims=x_lims,
                                 x_label=x_label,
                                 y_label=y_label,
                                 legend=legend,
                                 legend_loc=legend_loc,
                                 title=title,
                                 ax=slider.ax,
                                 trans=trans,
                                 linestyle=linestyle,
                                 marker=marker)

        slider.slider.on_changed(update_slider)

        return slider

    def plot_composition_slider(self, y_vars=None,
                                x_var='r',
                                x_scale=None, y_scale='linear',
                                x_lims=None, y_lims=None,
                                x_factor=None, y_factor=None,
                                x_label=None, y_label=None,
                                legend=True, legend_loc=None,
                                title=True,
                                trans=True,
                                ):
        """Plot interactive slider of isotope composition

        parameters
        ----------
        y_vars : [str] or 'all'
        x_var : str
        y_scale : 'log' or 'linear'
        x_scale : 'log' or 'linear'
        x_lims : [min, max]
        y_lims : [min, max]
        x_factor : float
        y_factor : float
        x_label : str
        y_label : str
        legend : bool
        legend_loc : str or int
        title : bool
        trans : bool
            plot helmholtz transitions
        """
        def update_slider(chk):
            chk = int(chk)

            profile = self.profiles.sel(chk=chk)
            x = profile[x_var]

            slider.update_trans_lines(chk=chk, x=x, y=y_lims)

            for y_var in y_vars:
                y = profile[y_var]
                slider.update_ax_line(x=x, y=y, y_var=y_var)

            title_str = self._get_title(chk=chk)
            plot.set_title(title_str=title_str)
            slider.fig.canvas.draw_idle()

        # ----------------
        self._check_composition()
        self._check_trans(trans=trans)
        x_factor, y_factor = self._check_factors(x_var=x_var,
                                                 y_var='X',
                                                 x_factor=x_factor,
                                                 y_factor=y_factor)

        if y_vars is None:
            y_vars = self.config.plotting('isotopes')
        elif y_vars == 'all':
            y_vars = [x.strip() for x in self.config.profiles('isotopes')]
            y_vars += ['ye']
        if y_lims is None:
            y_lims = self.config.ax_lims('X')

        slider = FlashSlider(y_vars=y_vars,
                             chk_table=self.chk_table,
                             trans=trans,
                             trans_dens=self.trans_dens,
                             x_factor=x_factor,
                             y_factor=y_factor)

        plot = self.plot_composition(chk=self.chk_table.index[-1],
                                     x_var=x_var,
                                     y_vars=y_vars,
                                     x_scale=x_scale,
                                     y_scale=y_scale,
                                     x_factor=x_factor,
                                     y_factor=y_factor,
                                     y_lims=y_lims,
                                     x_lims=x_lims,
                                     x_label=x_label,
                                     y_label=y_label,
                                     legend=legend,
                                     legend_loc=legend_loc,
                                     title=title,
                                     ax=slider.ax,
                                     trans=trans)

        slider.slider.on_changed(update_slider)

        return slider

    # =======================================================
    #                      Plotting Tools
    # =======================================================
    def _get_trans_xy(self, chk, trans_key, x, y):
        """Return x, y points of transition line, for given x-axis variable

        parameters
        ----------
        chk : int
        trans_key : str
        x : []
        y : []
        """
        trans_idx = self.chk_table.loc[chk, f'{trans_key}_i']

        trans_x = np.array([x[trans_idx], x[trans_idx]])
        trans_y = np.array([np.min(y), np.max(y)])

        return trans_x, trans_y

    def _get_title(self, chk, title_str=None):
        """Get title string

        Parameters
        ----------
        chk : int
        title_str : str
        """
        if (title_str is None) and (chk is not None):
            timestep = self.timesteps.loc[chk, 'time'] - self.bounce['time']
            title_str = f't = {timestep:.3f} s'

        return title_str

    def _plot_trans_lines(self, x, y, plot, chk, linewidth=1):
        """Add transition line to axis

        parameters
        ----------
        x : []
        y : []
        plot : Plotter
        chk : int
        linewidth : float
        """
        for trans_key in self.trans_dens:
            trans_x, trans_y = self._get_trans_xy(chk=chk,
                                                  trans_key=trans_key,
                                                  x=x,
                                                  y=y)
            plot.plot(x=trans_x,
                      y=trans_y,
                      linestyle='--',
                      marker='',
                      color='k',
                      linewidth=linewidth)

    def _check_factors(self, x_var, y_var, x_factor, y_factor):
        """Check if factors provided, otherwise use default

        Parameters
        ----------
        x_var : str
        y_var : str
        x_factor : float
        y_factor : float
        """
        if x_factor is None:
            x_factor = self.config.factor(x_var)
        if y_factor is None:
            y_factor = self.config.factor(y_var)

        return x_factor, y_factor

    # =======================================================
    #                   Convenience
    # =======================================================
    def _check_composition(self):
        """Check if isotopes defined
        """
        if len(self.config.profiles('isotopes')) == 0:
            raise AttributeError('No isotopes found. '
                                 'Are you sure this model use a network?')

    def _check_trans(self, trans):
        """Check if transition dens defined

        Parameters
        ----------
        trans : bool
        """
        if trans:
            if self.trans_dens is None:
                raise AttributeError('No trans_dens found. '
                                     'Are you sure this a hybrid-EOS model?')

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
