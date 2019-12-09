# flashbang
Python tools for extracting/analysing/plotting 1D core-collapse supernova models from [FLASH](http://flash.uchicago.edu/site/flashcode/).

# Dependencies
* numpy
* matplotlib
* yt
* h5py
* pyarrow
* pandas
* astropy
* scipy
You should already be using something like [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment) to manage separate python environments. In which case installing dependencies is as simple as `pip install numpy`, etc..

# Setup
Set these shell environment variables (in your `.bashrc` etc.):
* `FLASHBANG` - path to this code, e.g. `export FLASHBANG=${HOME}/codes/flashbang`
* `FLASH_MODELS` - path to FLASH models, e.g. `export FLASH_MODELS=${HOME}/BANG/runs`

In order to import with ipython etc., append to your python path: `export PYTHONPATH=${FLASHBANG}:${PYTHONPATH}`

# Getting Started
Everything is structured around the `Simulation` class, located in `flashbang/simulation.py`. This object represents a single FLASH simulation, and is intended to facilitate the loading/analysing/plotting of the model data.

`flashbang` generally assumes your model directory is structured like so:
```
$FLASH_MODELS
│
└───flash_model_1
│   │   run.dat
│   │   run.log
│   │   ...
│   │
│   └───output
│       │   run_hdf5_chk_0000
│       │   run_hdf5_chk_0001
│       │   ...
│
│___flash_model_2
│   ...
```

You can then initialise the Simulation object in python using:
```
import flashbang

sim = flashbang.simulation.Simulation(model='flash_model_1', 
                                      run='run', 
                                      output_dir='output')
```
**Warning:** If your model has more than a few `chk` files (like >10), this could take a long time (~2 sec per file). In that case use the arg `load_all=False` to skip loading for now. Look at `flashbang/scripts/extract_profiles.py` to "pre-load" a large number of `chk` files with multithreading (more detail to come soon...)

Where `model` is the name of the model directory, `run` is the prefix used in the output filenames (defaults to `'run'`), and `output` is the name of the output directory containing the `chk` and `plt` files (defaults to `'output'`).

# (To be continued...)
* config files (look in `flashbang/config/`)
* plotting
* jupyter notebook tutorial
