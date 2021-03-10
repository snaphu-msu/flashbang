"""
Script to extract chk profiles en-masse using multithreading

Usage:
    python extract_profiles <run> <model> <model_set>
"""
import sys
import multiprocessing as mp
import time

# flashbang
from flashbang import load_save
from flashbang.tools import str_to_bool
from flashbang.config import Config


def main(run, model, model_set,
         multithread=True,
         reload=False,
         save=True,
         config='default',
         threads=4):
    """
    Parameters
    ----------
    run : str
    model : str
    model_set : str
    multithread : bool
    reload : bool
    save : bool
    config : str
    threads : int
    """
    t0 = time.time()

    multithread = str_to_bool(multithread)
    reload = str_to_bool(reload)
    save = str_to_bool(save)
    threads = int(threads)
    config = Config(name=str(config))

    chk_list = load_save.find_chk(run=run, model=model, model_set=model_set)

    # Extract dat file
    load_save.get_dat(run=run,
                      model=model,
                      model_set=model_set,
                      config=config,
                      reload=reload,
                      save=save)

    # Extract profiles
    if multithread:
        args = []
        for chk in chk_list:
            args.append((chk, run, model, model_set, reload, save, config))

        with mp.Pool(processes=threads) as pool:
            pool.starmap(extract_profiles, args)
    else:
        for chk in chk_list:
            extract_profiles(chk=chk,
                             run=run,
                             model=model,
                             model_set=model_set,
                             reload=reload,
                             save=save,
                             config=config)

    # Combine all profiles
    load_save.get_multiprofile(run=run,
                               model=model,
                               model_set=model_set,
                               chk_list=chk_list,
                               reload=reload,
                               save=save)

    # Get timesteps
    load_save.get_timesteps(run=run,
                            model=model,
                            model_set=model_set,
                            chk_list=chk_list,
                            reload=reload,
                            save=save)

    t1 = time.time()
    print(f'Time taken: {t1-t0:.2f} s')


def extract_profiles(chk,
                     run,
                     model,
                     model_set,
                     reload,
                     save,
                     config):
    """Function for multithread pool
    """
    load_save.get_profile(chk=chk,
                          run=run,
                          model=model,
                          model_set=model_set,
                          reload=reload,
                          save=save,
                          config=config)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Parameters:'
              + '\n1. run'
              + '\n2. model'
              + '\n3. model_set'
              )
        sys.exit(0)
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3],
             **dict(arg.split('=') for arg in sys.argv[4:]))
