import matplotlib.pyplot as plt

# flashbang
from ..config import Config


class Plotter:
    def __init__(self,
                 ax=None,
                 config=None,
                 x_var=None, y_var=None,
                 x_scale=None, y_scale=None,
                 x_label=None, y_label=None,
                 x_lims=None, y_lims=None,
                 x_factor=None, y_factor=None,
                 legend=False, legend_loc=None,
                 title=False, title_str=None,
                 linestyle=None, marker=None,
                 linewidth=None,
                 figsize=None,
                 set_all=False,
                 verbose=True,
                 ):
        """Generalized plotter for handling axis properties

        Parameters
        ----------
        ax : pyplot Axis
        config : str or Config
        y_var : str
        x_var : str
        y_scale : one of ('log', 'linear')
        x_scale : one of ('log', 'linear')
        x_label : str
        y_label : str
        x_lims : [min, max]
        y_lims : [min, max]
        x_factor : float
        y_factor : float
        legend : bool
        legend_loc : str or int
        title : bool
        title_str : str
        figsize : []
        set_all : bool
        verbose : bool
        """
        self.x_var = x_var
        self.y_var = y_var
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.x_label = x_label
        self.y_label = y_label
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.legend = legend
        self.legend_loc = legend_loc
        self.title = title
        self.title_str = title_str
        self.linestyle = linestyle
        self.marker = marker
        self.linewidth = linewidth
        self.figsize = figsize
        self.verbose = verbose

        self.fig, self.ax = self.check_ax(ax=ax)
        self.config = self.check_config(config=config)
        self.check_properties()

        if set_all:
            self.set_all()

    # =======================================================
    #                      Setup
    # =======================================================
    def check_ax(self, ax):
        """Check ax provided, create if needed

        Returns : fig, ax
        """
        if ax is None:
            return plt.subplots(figsize=self.figsize)
        else:
            return None, ax

    def check_config(self, config):
        """Check config provided, load if needed

        Returns : Config
        """
        if (type(config) is str) or (config is None):
            return Config(name=config, verbose=self.verbose)
        elif type(config) is Config:
            return config
        else:
            raise TypeError(f'config must be string or Config')

    def check_properties(self):
        """Check provided plot properties, fallback on config
        """
        # labels
        if self.x_label is None:
            self.x_label = self.config.ax_label(self.x_var)
        if self.y_label is None:
            self.y_label = self.config.ax_label(self.y_var)

        # scales
        if self.x_scale is None:
            self.x_scale = self.config.ax_scale(self.x_var)
        if self.y_scale is None:
            self.y_scale = self.config.ax_scale(self.y_var)

        # factors
        if self.x_factor is None:
            self.x_factor = self.config.factor(self.x_var)
        if self.y_factor is None:
            self.y_factor = self.config.factor(self.y_var)

        # lims
        if self.x_lims is None:
            self.x_lims = self.config.ax_lims(self.x_var)
        if self.y_lims is None:
            self.y_lims = self.config.ax_lims(self.y_var)

        # legend
        if self.legend_loc is None:
            self.legend_loc = 3

    # =======================================================
    #                      Plot
    # =======================================================
    def plot(self, x, y,
             marker=None, linestyle=None,
             color=None,
             label=None,
             linewidth=None,
             **kwargs
             ):
        """Plot data on axis

        Parameters
        ----------
        x : []
        y : []
        marker : str
        linestyle : str
        color : str
        label : str
        linewidth : float
        **kwargs
            valid kwargs for ax.plot()
        """
        if marker is None:
            marker = self.marker
        if linestyle is None:
            linestyle = self.linestyle
        if linewidth is None:
            linewidth = self.linewidth

        self.ax.plot(x/self.x_factor,
                     y/self.y_factor,
                     marker=marker,
                     linestyle=linestyle,
                     label=label,
                     color=color,
                     linewidth=linewidth,
                     **kwargs)

    # =======================================================
    #                      Axis
    # =======================================================
    def set_all(self):
        """Set all axis properties
        """
        self.set_labels()
        self.set_scales()
        self.set_lims()
        self.set_title()
        self.set_legend()

    def set_labels(self):
        """Set axis labels
        """
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)

    def set_scales(self):
        """Set axis scales (log or linear)
        """
        self.ax.set_xscale(self.x_scale)
        self.ax.set_yscale(self.y_scale)

    def set_lims(self):
        """Set axis limits (min, max)
        """
        if self.x_lims is not None:
            self.ax.set_xlim(self.x_lims)
        if self.y_lims is not None:
            self.ax.set_ylim(self.y_lims)

    def set_title(self, title_str=None):
        """Set axis title
        """
        if title_str is None:
            title_str = self.title_str

        if self.title:
            self.ax.set_title(title_str)

    def set_legend(self):
        """Set axis legend
        """
        if self.legend:
            self.ax.legend(loc=self.legend_loc)
