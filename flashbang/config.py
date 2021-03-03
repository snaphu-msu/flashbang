"""Config class
"""
import os
import configparser
import ast

# flashbang
from .paths import config_filepath
from .tools import printv


class ConfigError(Exception):
    pass


class Config:
    def __init__(self,
                 name=None,
                 verbose=True):
        """Holds and returns config values

        Parameters
        ----------
        name : str or None
            name of config to load, e.g. 'stir'
            if None, uses 'default'
        verbose : bool
        """
        if name is None:
            self.name = 'default'
        else:
            self.name = name

        self.verbose = verbose
        self.config = load_config_file(name=self.name, verbose=self.verbose)

        # override any options from plotting.ini
        plot_config = load_config_file(name='plotting', verbose=self.verbose)
        plot_config['plotting'].update(self.config['plotting'])
        self.config.update(plot_config)

    # ===============================================================
    #                  Accessing Properties
    # ===============================================================
    def profiles(self, var):
        """Get profiles property
        """
        conf = self.config['profiles']

        if var == 'all':
            return conf['params'] + conf['isotopes']
        elif var not in conf:
            raise ConfigError(f"'{var}' not a valid profiles property")
        else:
            return conf[var]

    def dat(self, var):
        """Get dat property
        """
        if var not in ['columns']:
            raise ConfigError(f"'{var}' not a valid dat property")
        else:
            return self.config['dat_columns']

    def trans(self, var):
        """Get transitions property
        """
        conf = self.config['transitions']

        if var not in conf:
            raise ConfigError(f"'{var}' not a valid trans property")
        else:
            return conf[var]

    def tracers(self, var):
        """Get tracers property
        """
        conf = self.config['tracers']

        if var not in conf:
            raise ConfigError(f"'{var}' not a valid tracers property")
        else:
            return conf[var]

    def plotting(self, var):
        """Get plotting property
        """
        conf = self.config['plotting']

        if var not in conf:
            raise ConfigError(f"'{var}' not a valid plotting property")
        else:
            return conf[var]

    def ax_scale(self, var):
        """Get axis scale for given var, default to 'linear'

        Returns : 'log' or 'linear'
        """
        if var in self.plotting('ax_scales')['log']:
            return 'log'
        else:
            return 'linear'

    def ax_lims(self, var):
        """Get axis limits for given var

        Returns : [min, max]
        """
        return self.plotting('ax_lims').get(var)

    def ax_label(self, var):
        """Get axis label for given var

        Returns : str
        """
        return self.plotting('labels').get(var, var)

    def check_trans(self, trans):
        """Gets trans option from config if not specified, default to False

        Returns : bool
        """
        if trans is None:
            trans = self.plotting('options').get('trans', False)

        return trans

    def factor(self, var):
        """Get scaling factor, default to 1.0

        Returns float
        """
        return self.plotting('factors').get(var, 1.0)


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
