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
                     linestyle='-',
                     title_str=None):
        """Plot comparison profile
        """
        fig, ax = plot_tools.setup_fig(ax=ax)

        for model, sim in self.sims.items():
            sim.plot_profile(chk=chk, y_var=y_var, x_var=x_var,
                             y_scale=y_scale, x_scale=x_scale,
                             marker=marker, trans=trans,
                             title=title, linestyle=linestyle,
                             title_str=title_str,
                             ylims=ylims, xlims=xlims, y_factor=y_factor,
                             legend=False, ax=ax, label=model)
        ax.legend()
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
