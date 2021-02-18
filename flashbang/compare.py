"""Compare multiple simulations
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# flashbang
from . import simulation
from . import plot_tools
from . import load_save


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
        """
        self.sims = {}
        self.verbose = verbose
        self.config = load_save.load_config(config, verbose=self.verbose)

        for i, model in enumerate(models):
            self.sims[model] = simulation.Simulation(run=runs[i],
                                                     model=model,
                                                     model_set=model_sets[i],
                                                     config=config)

    # =======================================================
    #                      Plot
    # =======================================================
    def plot_profile(self, chk, y_var,
                     x_var='r',
                     y_scale=None, x_scale=None,
                     ylims=None, xlims=None,
                     y_factor=1,
                     ax=None,
                     marker=None,
                     trans=False,
                     title=True,
                     legend=True,
                     linestyle='-',
                     title_str=None,
                     data_only=False):
        """Plot profile comparison
        """
        fig, ax = plot_tools.setup_fig(ax=ax)

        for model, sim in self.sims.items():
            sim.plot_profile(chk=chk, y_var=y_var, x_var=x_var,
                             y_factor=y_factor, marker=marker, trans=trans,
                             linestyle=linestyle, ax=ax, label=model,
                             data_only=True)

        if not data_only:
            self._set_ax_all(ax, x_var=x_var, y_var=y_var, xlims=xlims, ylims=ylims,
                             x_scale=x_scale, y_scale=y_scale, chk=chk, title=title,
                             title_str=title_str, legend=legend)
        return fig

    def plot_dat(self,
                 y_var,
                 **kwargs):
        """Plot comparison dat
        """
        fig, ax = plt.subplots()

        for model, sim in self.sims.items():
            sim.plot_dat(y_var=y_var, ax=ax, label=model,
                         **kwargs)
        ax.legend()
        return fig

    # =======================================================
    #                      Plotting Tools
    # =======================================================
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
        # TODO: use chk_table from master model
        if title:
            if (title_str is None) and (chk is not None):
                # timestep = self.chk_table.loc[chk, 'time'] - self.bounce_time
                dt = self.config['plotting']['scales']['chk_dt']
                timestep = dt * chk
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
        loc : int or str
        """
        if legend:
            ax.legend(loc=loc)

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
