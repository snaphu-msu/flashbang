"""Compare multiple simulations

The Comparison object represents a collection of 1D FLASH models.
It loads multiple models for plotting.
For details on model attributes and nomencclature, see simulation.py
"""
import numpy as np
from matplotlib.widgets import Slider

# flashbang
from . import simulation
from .plotting import plot_tools
from .plotting.plotter import Plotter
from .plotting.slider import FlashSlider
from . import tools
from .config import Config


class Comparison:
    """Object for holding multiple models to compare
    """
    def __init__(self,
                 runs,
                 models,
                 model_sets,
                 config=None,
                 verbose=True):
        """
        Parameters
        ----------
        runs : [str]
        models : [str]
        model_sets : [str]
        config : str
        verbose : bool
        """
        self.sims = {}
        self.verbose = verbose
        self.config = Config(name=config, verbose=self.verbose)

        n_models = len(models)
        self.runs = tools.ensure_sequence(runs, n_models)
        self.models = models
        self.model_sets = tools.ensure_sequence(model_sets, n_models)
        self.load_models(config=config)

        self.baseline = models[0]
        self.baseline_sim = self.sims[self.baseline]

    # =======================================================
    #                      Loading
    # =======================================================
    def load_models(self, config):
        """Load all models

        Parameters
        ----------
        config : str
        """
        for i, model in enumerate(self.models):
            self.sims[model] = simulation.Simulation(run=self.runs[i],
                                                     model=model,
                                                     model_set=self.model_sets[i],
                                                     config=config,
                                                     verbose=self.verbose)

    # =======================================================
    #                      Plot
    # =======================================================
    def plot_profile(self, chk, y_var,
                     x_var='r',
                     x_scale=None, y_scale=None,
                     x_lims=None, y_lims=None,
                     x_factor=None, y_factor=None,
                     x_label=None, y_label=None,
                     legend=True, legend_loc=None,
                     title=True, title_str=None,
                     marker=None, linestyle=None,
                     ax=None,
                     trans=None,
                     data_only=False):
        """Plot profile comparison

        Returns : fig

        Parameters
        ----------
        chk : int
        y_var : str
            variable to plot on y-axis (from Simulation.profile)
        x_var : str
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
        trans : bool
        data_only : bool
            only plot data, neglecting all titles/labels/scales
        """
        title_str = self._get_title(chk=chk, title_str=title_str)

        plot = Plotter(ax=ax, config=self.config,
                       x_var=x_var, y_var=y_var,
                       x_lims=x_lims, y_lims=y_lims,
                       x_scale=x_scale, y_scale=y_scale,
                       x_label=x_label, y_label=y_label,
                       x_factor=x_factor, y_factor=y_factor,
                       title=title, title_str=title_str,
                       legend=legend, legend_loc=legend_loc,
                       verbose=self.verbose)

        for model, sim in self.sims.items():
            sim.plot_profile(chk=chk, y_var=y_var, x_var=x_var,
                             x_factor=x_factor, y_factor=y_factor,
                             marker=marker, linestyle=linestyle,
                             trans=trans if model == self.baseline else False,
                             ax=plot.ax, label=model,
                             data_only=True)

        if not data_only:
            plot.set_all()

        return plot

    def plot_dat(self, y_var,
                 x_scale=None, y_scale=None,
                 x_lims=None, y_lims=None,
                 x_factor=None, y_factor=None,
                 x_label=None, y_label=None,
                 legend=True, legend_loc=None,
                 title=True, title_str=None,
                 ax=None,
                 linestyle=None, marker=None,
                 zero_time=True,
                 data_only=False):
        """Plot time-dependent datfile comparison

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
        ax : Axes
        linestyle : str
        marker : str
        zero_time : bool
        data_only : bool
        """
        plot = Plotter(ax=ax, config=self.config,
                       x_var='time', y_var=y_var,
                       x_lims=x_lims, y_lims=y_lims,
                       x_scale=x_scale, y_scale=y_scale,
                       x_label=x_label, y_label=y_label,
                       title=title, title_str=title_str,
                       legend=legend, legend_loc=legend_loc,
                       verbose=self.verbose)

        for model, sim in self.sims.items():
            sim.plot_dat(y_var=y_var, ax=plot.ax, label=model,
                         x_factor=x_factor, y_factor=y_factor,
                         marker=marker, zero_time=zero_time,
                         linestyle=linestyle, data_only=True)

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
                            x_factor=1, y_factor=1,
                            x_label=None, y_label=None,
                            legend=True, legend_loc=None,
                            title=True,
                            trans=None,
                            linestyle='-',
                            marker=''):
        """Plot interactive profile comparison

        Returns : fig, slider

        Parameters
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
        legend : bool
        legend_loc : str or int
        title : bool
        trans : bool
            plot helmholtz transitions
        linestyle : str
        marker : str
        """
        def update_slider(chk):
            chk = int(chk)

            if trans:
                self._update_slider_trans(chk=chk, x_var=x_var, y_var=y_var,
                                          x_factor=x_factor, y_factor=y_factor,
                                          slider=slider)

            self._update_slider_profiles(chk=chk, x_var=x_var, y_var=y_var,
                                         x_factor=x_factor, y_factor=y_factor,
                                         slider=slider)

            # self._set_ax_title(ax=profile_ax, chk=chk, title=title)
            slider.fig.canvas.draw_idle()

        # ----------------
        if trans is None:
            trans = self.config.trans('plot')

        y_vars = []
        for i in range(len(self.models)):
            y_vars += [f'{y_var}_{i}']

        chk_min, chk_max = self._get_slider_chk()
        slider = self._setup_slider(y_vars=y_vars, trans=trans)

        plot = self.plot_profile(chk=chk_max,
                                 y_var=y_var, x_var=x_var,
                                 y_scale=y_scale, x_scale=x_scale,
                                 y_lims=y_lims, x_lims=x_lims,
                                 x_factor=x_factor, y_factor=y_factor,
                                 x_label=x_label, y_label=y_label,
                                 legend=False, legend_loc=legend_loc,
                                 title=title,
                                 ax=slider.ax,
                                 trans=False,
                                 linestyle=linestyle,
                                 marker=marker)

        if trans:
            self._plot_trans_lines(x_var=x_var, y_var=y_var,
                                   x_factor=x_factor, y_factor=y_factor,
                                   plot=plot, chk=chk_max)

        # self._set_ax_legend(ax=profile_ax, legend=legend)
        slider.slider.on_changed(update_slider)

        return slider, plot

    # =======================================================
    #                      Plotting Tools
    # =======================================================
    def _get_baseline_xy(self, chk, x_var, y_var, x_factor, y_factor):
        """Update trans lines on slider plot

        Parameters
        ----------
        chk : int
        x_var : str
        y_var : str
        x_factor : float
        y_factor : float
        """
        profile = self.baseline_sim.profiles.sel(chk=chk)

        x = profile[x_var] / x_factor
        y = profile[y_var] / y_factor

        return x, y

    def _get_trans_xy(self, chk, trans_key, x, y):
        """Return x, y points of transition line, for given x-axis variable

        parameters
        ----------
        chk : int
        trans_key : str
        x : []
        y : []
        """
        trans_idx = self.baseline_sim.chk_table.loc[chk, f'{trans_key}_i']

        trans_x = np.array([x[trans_idx], x[trans_idx]])
        trans_y = np.array([np.min(y), np.max(y)])

        return trans_x, trans_y

    def _plot_trans_lines(self, x_var, y_var, x_factor, y_factor,
                          plot, chk, linewidth=1):
        """Add transition line to axis

        parameters
        ----------
        x_var : str
        y_var : str
        x_factor : float
        y_factor : float
        plot : Plotter
        chk : int
        linewidth : float
        """
        x, y = self._get_baseline_xy(chk=chk, x_var=x_var, y_var=y_var,
                                     x_factor=x_factor, y_factor=x_factor)

        for trans_key in self.baseline_sim.trans_dens:
            trans_x, trans_y = self._get_trans_xy(chk=chk, trans_key=trans_key,
                                                  x=x, y=y)
            plot.plot(trans_x, trans_y, linestyle='--', marker='',
                      color='k', linewidth=linewidth)

    def _get_title(self, chk, title_str):
        """Get title string

        Parameters
        ----------
        chk : int
        title_str : str
        """
        if (title_str is None) and (chk is not None):
            timestep = self.baseline_sim.timesteps.loc[chk, 'time']
            bounce = self.baseline_sim.bounce['time']

            timestep = timestep - bounce
            title_str = f't = {timestep:.3f} s'

        return title_str

    # =======================================================
    #                      Slider Tools
    # =======================================================
    def _get_slider_chk(self):
        """Return largest chk range common to all models
        """
        mins = []
        maxes = []

        for sim in self.sims.values():
            mins += [sim.chk_table.index.min()]
            maxes += [sim.chk_table.index.max()]

        chk_min = max(mins)
        chk_max = min(maxes)
        return chk_min, chk_max

    def _setup_slider(self, y_vars, trans):
        """Return slider fig
        """
        chk_min, chk_max = self._get_slider_chk()
        chk_table = self.baseline_sim.chk_table.loc[chk_min:chk_max]

        slider = FlashSlider(y_vars=y_vars,
                             chk_table=chk_table,
                             trans=trans,
                             trans_dens=self.baseline_sim.trans_dens)
        return slider

    def _get_ax_lines(self, ax, trans):
        """Return dict of plot lines for each model

        Parameters
        ----------
        ax : Axis
        trans : bool
        """
        lines = {}
        trans_offset = 0
        lines[self.baseline] = ax.lines[0]
        sim_0 = self.sims[self.baseline]

        if trans:
            trans_offset = len(sim_0.trans_dens)
            for i, trans_key in enumerate(sim_0.trans_dens):
                lines[trans_key] = ax.lines[1+i]

        for i, model in enumerate(self.models[1:]):
            lines[model] = ax.lines[1+trans_offset+i]

        return lines

    def _update_slider_profiles(self, chk, x_var, y_var,
                                x_factor, y_factor, slider):
        """Update profile lines on slider plot

        Parameters
        ----------
        chk : int
        x_var : str
        y_var : str
        x_factor : float
        y_factor : float
        slider : FlashSlider
        """
        for i, sim in enumerate(self.sims.values()):
            profile = sim.profiles.sel(chk=chk)
            x = profile[x_var] / x_factor
            y = profile[y_var] / y_factor

            slider.update_ax_line(x=x, y=y, y_var=f'{y_var}_{i}')

    def _update_slider_trans(self, chk, x_var, y_var,
                             x_factor, y_factor, slider):
        """Update trans lines on slider plot

        Parameters
        ----------
        chk : int
        x_var : str
        y_var : str
        x_factor : float
        y_factor : float
        slider : FlashSlider
        """
        x, y = self._get_baseline_xy(chk=chk, x_var=x_var, y_var=y_var,
                                     x_factor=x_factor, y_factor=y_factor)
        slider.update_trans_lines(chk=chk, x=x, y=y)

