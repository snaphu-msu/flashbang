import os

bangpy_path = os.environ['BANGPY']
models_path = os.environ['BANG_MODELS']


def config_path(name='config'):
    """Returns path to config file
    """
    return os.path.join(bangpy_path, f'{name}.ini')