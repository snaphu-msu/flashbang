"""Compare multiple simulations

The Comparison object represents a collection of 1D FLASH models.
It loads multiple models for plotting.
For details on model attributes and nomencclature, see simulation.py
"""
import numpy as np
from matplotlib.widgets import Slider

# flashbang
from . import simulation
from . import plot_tools
from . import load_save
from . import tools


class Comparison:
    """Object for holding multiple models to compare
    """
    def __init__(self,
                 runs,
                 models,
                 model_sets,
                 config,
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
        self.config = load_save.load_config(config, verbose=self.verbose)

        n_models = len(models)
        self.baseline = models[0]
        self.runs = tools.ensure_sequence(runs, n_models)
        self.models = models
        self.model_sets = tools.ensure_sequence(model_sets, n_models)
        self.load_models(config=config)

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
                     xlims=None, ylims=None,
                     x_factor=1, y_factor=1,
                     ax=None,
                     marker=None,
                     trans=False,
                     title=True,
                     legend=True,
                     linestyle='-',
                     title_str=None,
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
        xlims : [min, max]
        ylims : [min, max]
        x_factor : float
            Divide x-values by this factor
        y_factor : float
            Divide y-values by this factor
        ax : Axes
        legend : bool
        trans : bool
        title : bool
        linestyle : str
        marker : str
        title_str : str
        data_only : bool
            only plot data, neglecting all titles/labels/scales
        """
        fig, ax = plot_tools.setup_fig(ax=ax)

        for model, sim in self.sims.items():
            sim.plot_profile(chk=chk, y_var=y_var, x_var=x_var,
                             x_factor=x_factor, y_factor=y_factor,
                             marker=marker,
                             trans=trans if model == self.baseline else False,
                             linestyle=linestyle, ax=ax, label=model,
                             data_only=True)

        if not data_only:
            self._set_ax_all(ax, x_var=x_var, y_var=y_var, xlims=xlims, ylims=ylims,
                             x_scale=x_scale, y_scale=y_scale, chk=chk, title=title,
                             title_str=title_str, legend=legend)
        return fig

    def plot_dat(self, y_var,
                 x_scale=None, y_scale=None,
                 xlims=None, ylims=None,
                 x_factor=1, y_factor=1,
                 ax=None,
                 linestyle='-',
                 marker='',
                 legend=True,
                 zero_time=True,
                 title_str=None,
                 data_only=False):
        """Plot time-dependent datfile comparison

        parameters
        ----------
        y_var : str
        x_scale : 'log' or 'linear'
        y_scale : 'log' or 'linear'
        xlims : [min, max]
        ylims : [min, max]
        x_factor : float
        y_factor : float
        ax : Axes
        linestyle : str
        marker : str
        legend : bool
        zero_time : bool
        title_str : str
        data_only : bool
        """
        fig, ax = plot_tools.setup_fig(ax=ax)

        for model, sim in self.sims.items():
            sim.plot_dat(y_var=y_var, ax=ax, label=model,
                         x_scale=x_scale, y_scale=y_scale,
                         x_factor=x_factor, y_factor=y_factor,
                         marker=marker, zero_time=zero_time,
                         linestyle=linestyle,
                         data_only=True, legend=False)

        if not data_only:
            self._set_ax_all(ax, x_var='time', y_var=y_var, xlims=xlims,
                             ylims=ylims, x_scale=x_scale, y_scale=y_scale,
                             title_str=title_str, legend=legend, title=False)

        return fig

    # =======================================================
    #                      Sliders
    # =======================================================
    def plot_profile_slider(self, y_var,
                            x_var='r',
                            x_scale=None, y_scale=None,
                            xlims=None, ylims=None,
                            x_factor=1, y_factor=1,
                            trans=False,
                            title=True,
                            legend=True,
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
        trans : bool
            plot helmholtz transitions
        title : bool
        xlims : [min, max]
        ylims : [min, max]
        x_factor : float
        y_factor : float
        legend : bool
        linestyle : str
        marker : str
        """
        def update_slider(chk):
            chk = int(chk)

            if trans:
                self._update_trans_lines(chk=chk, sim=self.sims[self.baseline],
                                         x_var=x_var, y_var=y_var,
                                         x_factor=x_factor, y_factor=y_factor,
                                         lines=lines)

            self._update_profile_lines(chk=chk, x_var=x_var, y_var=y_var,
                                       x_factor=x_factor, y_factor=y_factor,
                                       lines=lines)

            self._set_ax_title(ax=profile_ax, chk=chk, title=title)
            fig.canvas.draw_idle()

        # ----------------
        fig, profile_ax, slider = self._setup_slider()
        chk_min, chk_max = self._get_slider_chk()

        self.plot_profile(chk=chk_max,
                          y_var=y_var, x_var=x_var,
                          y_scale=y_scale, x_scale=x_scale,
                          ylims=ylims, xlims=xlims,
                          x_factor=x_factor, y_factor=y_factor,
                          ax=profile_ax, legend=False,
                          trans=trans, title=title,
                          linestyle=linestyle,
                          marker=marker)

        self._set_ax_legend(ax=profile_ax, legend=legend)
        lines = self._get_ax_lines(ax=profile_ax, trans=trans)
        slider.on_changed(update_slider)

        return fig, slider

    # =======================================================
    #                      Plotting Tools
    # =======================================================
    def _set_ax_all(self, ax,
                    x_var, y_var,
                    x_scale, y_scale,
                    xlims, ylims,
                    title,
                    legend,
                    loc=None,
                    chk=None,
                    title_str=None):
        """Set all axis properties

        Parameters
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

    def _set_ax_scales(self, ax, x_var, y_var, x_scale, y_scale):
        """Set axis scales (linear, log)

        Parameters
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

        Parameters
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
                timestep = dt * chk
                title_str = f't = {timestep:.3f} s'

            ax.set_title(title_str)

    def _set_ax_lims(self, ax, xlims, ylims):
        """Set x and y axis limits

        Parameters
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

        Parameters
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

        Parameters
        ----------
        ax : Axes
        legend : bool
        loc : int or str
        """
        if legend:
            ax.legend(loc=loc)

    def _get_trans_xy(self, chk, sim, trans_key, x, y):
        """Return x, y points of transition line, for given x-axis variable

        parameters
        ----------
        chk : int
        sim : Simulation
        trans_key : str
        x : []
        y : []
        """
        trans_idx = sim.chk_table.loc[chk, f'{trans_key}_i']

        trans_x = np.array([x[trans_idx], x[trans_idx]])
        trans_y = np.array([np.min(y), np.max(y)])

        return trans_x, trans_y

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

    def _setup_slider(self):
        """Return slider fig
        """
        fig, profile_ax, slider_ax = plot_tools.setup_slider_fig()
        chk_min, chk_max = self._get_slider_chk()
        slider = Slider(slider_ax, 'chk', chk_min, chk_max, valinit=chk_max, valstep=1)

        return fig, profile_ax, slider

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

    def _update_ax_line(self, x, y, line):
        """Update x,y lines on slider plot

        Parameters
        ----------
        x : ndarray
        y : ndarray
        line : Axis.line
        """
        line.set_xdata(x)
        line.set_ydata(y)

    def _update_profile_lines(self, chk, x_var, y_var,
                              x_factor, y_factor, lines):
        """Update profile lines on slider plot

        Parameters
        ----------
        chk : int
        x_var : str
        y_var : str
        x_factor : float
        y_factor : float
        lines : {var: Axis.line}
        """
        for model, sim in self.sims.items():
            profile = sim.profiles.sel(chk=chk)
            x = profile[x_var] / x_factor
            y = profile[y_var] / y_factor

            self._update_ax_line(x=x, y=y, line=lines[model])

    def _update_trans_lines(self, chk, sim, x_var, y_var,
                            x_factor, y_factor, lines):
        """Update trans lines on slider plot

        Parameters
        ----------
        chk : int
        sim : Simulation
        x_var : str
        y_var : str
        x_factor : float
        y_factor : float
        lines : {var: Axis.line}
        """
        profile = sim.profiles.sel(chk=chk)
        x = profile[x_var] / x_factor
        y = profile[y_var] / y_factor

        for trans_key in sim.trans_dens:
            trans_x, trans_y = self._get_trans_xy(chk=chk, sim=sim, trans_key=trans_key,
                                                  x=x, y=y)
            self._update_ax_line(x=trans_x, y=trans_y, line=lines[trans_key])
