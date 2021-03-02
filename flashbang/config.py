"""Config class
"""
import os
import configparser
import ast

# flashbang
from .paths import config_filepath
from .tools import printv


# ===============================================================
#                      Config
# ===============================================================
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
        
        self.load()

    def load(self):
        """
        """
        self.config = load_config_file(name=self.name, verbose=self.verbose)

        # override any options from plotting.ini
        plot_config = load_config_file(name='plotting', verbose=self.verbose)
        plot_config['plotting'].update(self.config['plotting'])
        self.config.update(plot_config)


# ===============================================================
#                      Functions
# ===============================================================
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
