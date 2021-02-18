"""Compare multiple simulations
"""
import matplotlib.pyplot as plt

from . import simulation


class Comparison:
    def __init__(self,
                 runs,
                 models,
                 model_sets,
                 config):
        self.sims = {}

        for i, model in enumerate(models):
            self.sims[model] = simulation.Simulation(run=runs[i],
                                                     model=model,
                                                     model_set=model_sets[i],
                                                     config=config)

    def plot_profile(self,
                     chk,
                     y_var,
                     **kwargs):
        """Plot comparison profile
        """
        fig, ax = plt.subplots()

        for model, sim in self.sims.items():
            sim.plot_profile(chk=chk, y_var=y_var, ax=ax, label=model,
                             **kwargs)
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
