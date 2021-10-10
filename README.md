# flashbang
Python tools for extracting/analysing/plotting 1D core-collapse supernova models from [FLASH](http://flash.uchicago.edu/site/flashcode/).

# Python Dependencies
* python 3.7
* astropy
* h5py
* matplotlib
* numpy
* pandas
* scipy
* xarray
* yt

Use the included `environment.yml` file to easily set up a working [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) environment with the necessary dependencies.

Simply run 

`conda env create -f environment.yml`

which will create a python environment called `flashbang`, which you can then activate with 

`conda activate flashbang`

# Setup
Set these shell environment variables (e.g. in your `.bashrc`):
* `FLASHBANG` - path to this code, e.g. `export FLASHBANG=${HOME}/codes/flashbang`
* `FLASH_MODELS` - path to FLASH models, e.g. `export FLASH_MODELS=${HOME}/BANG/runs`

In order to import with ipython etc., append to your python path: `export PYTHONPATH=${FLASHBANG}:${PYTHONPATH}`

# Getting Started
Everything is structured around the `Simulation` class, located in `flashbang/simulation.py`. This object represents a single FLASH simulation, and is intended to facilitate the loading/analysing/plotting of the model data.

`flashbang` assumes your model directories are structured like so:
```
$FLASH_MODELS
│
└───model_set_1
|   |
|   └───flash_model_1
|   │   │   run_1.dat
|   │   │   run_1.log
|   │   │   ...
|   │   │
|   │   └───output
|   │       │   run_1_hdf5_chk_0000
|   │       │   run_1_hdf5_chk_0001
|   │       │   ...
|
|   │___flash_model_2
|   │   ...
```

For this example, you can initialise the Simulation object in python using:
```
import flashbang

sim = flashbang.simulation.Simulation(model='flash_model_1', 
                                      run='run_1', 
                                      model_set='model_set_1',
                                      output_dir='output')
```
Where `model` is the name of the model directory, `run` is the prefix used in the output filenames, `model_set` is the name of the directory containing `model`, and `output` is the name of the directory containing the `chk` and `plt` files (defaults to `'output'`).

**Warning:** If your model has more than a few `chk` files this could take a long time (~2 sec per file). In that case you can use the arg `load_all=False` to skip loading for now. Look at `flashbang/scripts/extract_profiles.py` to pre-extract a large number of `chk` files with multithreading (more detail to come soon...)

# (To be continued...)
* preloading files with `extract_profiles`
* config files (look in `flashbang/config/`)
* plotting
* jupyter notebook tutorial
