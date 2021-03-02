"""Config class
"""
import os
import configparser
import ast

# flashbang
from .paths import config_filepath
from .tools import printv


class Config:
    def __init__(self,
                 name,
                 verbose=True):
        """Holds and returns config values

        Parameters
        ----------
        name : str
            name of config to load, e.g. 'stir'
        verbose : bool
        """
        self.config = None
        self.name = name
        self.verbose = verbose

        self.profiles = Property()
        self.tracers = Property()
        self.trans = Property()
        self.dat = Property()
        self.plotting = Property()
        self.paths = Property()

        self.load()
        self.extract()

    # ===============================================================
    #                      Loading
    # ===============================================================
    def load(self):
        """Load config files
        """
        self.config = load_config_file(name=self.name, verbose=self.verbose)

        # override any options from plotting.ini
        plot_config = load_config_file(name='plotting', verbose=self.verbose)
        plot_config['plotting'].update(self.config['plotting'])
        self.config.update(plot_config)

    def extract(self):
        """Extract config attributes from dict
        """
        self.profiles.params = self.config['profiles']['params']
        self.profiles.isotopes = self.config['profiles']['isotopes']
        self.profiles.derived = self.config['profiles']['derived_params']
        self.profiles.all_params = self.profiles.params + self.profiles.isotopes

        self.dat.columns = self.config['dat_columns']

        self.trans.dens = self.config['transitions']['dens']
        self.tracers.mass_grid = self.config['tracers']['mass_grid']
        self.tracers.params = self.config['tracers']['params']

        self.plotting.isotopes = self.config['plotting']['isotopes']
        self.plotting.labels = self.config['plotting']['labels']
        self.plotting.scales = self.config['plotting']['scales']
        self.plotting.ax_scales = self.config['plotting']['ax_scales']
        self.plotting.ax_lims = self.config['plotting']['ax_lims']
        self.plotting.options = self.config['plotting']['options']

        self.paths.output_dir = self.config['paths']['output_dir']
        self.paths.run_default = self.config['paths']['run_default']

    # ===============================================================
    #                      Accessing
    # ===============================================================
    def get_ax_lims(self, var):
        """Get axis limits for given var

        Returns : [min, max]
        """
        return self.plotting.ax_lims.get(var)

    def get_ax_label(self, var):
        """Get axis label for given var

        Returns : str
        """
        return self.plotting.labels.get(var, var)

    def check_trans(self, trans):
        """Gets trans option from config if not specified, default to False
        """
        if trans is None:
            trans = self.plotting.options.get('trans', False)

        return trans


class Property:
    """Dummy class to hold attributes"""
    pass


def load_config_file(name, verbose=True):
    """Load .ini config file and return as dict

    Returns : {}

    Parameters
    ----------
    name : str
    verbose : bool
    """
    filepath = config_filepath(name=name)
    printv(f'Loading config: {filepath}', verbose)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Config file not found: {filepath}')

    ini = configparser.ConfigParser()
    ini.read(filepath)

    config = {}
    for section in ini.sections():
        config[section] = {}
        for option in ini.options(section):
            config[section][option] = ast.literal_eval(ini.get(section, option))

    return config
