# flashbang
Python tools for extracting/analysing/plotting 1D Core-collapse supernova models from FLASH.

# Dependencies
* numpy
* matplotlib
* yt
* h5py
* pyarrow
* pandas
* astropy
* scipy

# Setup
Set these shell environment variables (in your `.bashrc` etc.):
* `FLASHBANG` - path to this code, e.g. `export FLASHBANG=${HOME}/codes/flashbang`
* `FLASH_MODELS` - path to FLASH models, e.g. `export FLASH_MODELS=${HOME}/BANG/runs`

In order to import with ipython etc., append to your python path: `export PYTHONPATH=${FLASHBANG}:${PYTHONPATH}`
