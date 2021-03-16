import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt


class FlashSlider:
    def __init__(self,
                 y_vars,
                 chk_table,
                 trans,
                 trans_dens,
                 x_factor=1,
                 y_factor=1,
                 ):
        """Handles a slider plot

        Parameters
        ----------
        y_vars : [str]
            List of y-variables being plotted
        chk_table : pd.DataFrame
            Table of chk values (see Simulation.chk_table)
        trans : bool
            Whether transition density lines are plotted
        trans_dens : {}
            names and values of transition densities
        x_factor : float
        y_factor : float
        """
        self.y_vars = y_vars
        self.chk_table = chk_table
        self.trans = trans
        self.trans_dens = trans_dens
        self.x_factor = x_factor
        self.y_factor = y_factor

        self.fig, self.ax, self.slider = self.setup()
        self.lines = None

    # =======================================================
    #                      Setup
    # =======================================================
    def setup(self, figsize=(8, 6)):
        """Setup slider fig

        Returns : fig, profile_ax, slider
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.65])
        slider_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05])

        chk_min = self.chk_table.index[0]
        chk_max = self.chk_table.index[-1]

        slider = Slider(slider_ax, 'chk', chk_min, chk_max, valinit=chk_max, valstep=1)

        return fig, ax, slider

    def get_ax_lines(self):
        """Return dict of labelled axis lines
        """
        lines = {}
        n_vars = len(self.y_vars)

        for i, y_var in enumerate(self.y_vars):
            lines[y_var] = self.ax.lines[i]

        if self.trans:
            for i, trans_key in enumerate(self.trans_dens):
                lines[trans_key] = self.ax.lines[n_vars+i]

        self.lines = lines

    # =======================================================
    #                      Plot
    # =======================================================
    def update_ax_line(self, x, y, y_var):
        """Update x,y line values

        Parameters
        ----------
        x : array
        y : array
        y_var : str
        """
        if self.lines is None:
            self.get_ax_lines()

        self.lines[y_var].set_xdata(x / self.x_factor)
        self.lines[y_var].set_ydata(y / self.y_factor)

    def update_trans_lines(self, chk, x, y):
        """Update trans line values on plot

        Parameters
        ----------
        chk : int
        x : []
        y : []
        """
        if self.trans:
            for trans_key in self.trans_dens:
                trans_x, trans_y = self.get_trans_xy(chk=chk,
                                                     trans_key=trans_key,
                                                     x=x, y=y)
                self.update_ax_line(x=trans_x, y=trans_y, y_var=trans_key)

    def get_trans_xy(self, chk, trans_key, x, y):
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
