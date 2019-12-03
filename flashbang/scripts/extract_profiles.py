import sys
import multiprocessing as mp

from flashbang import simulation, load_save

# =================================================================
# Script callable from terminal to extract model profiles
# Usage:
#   python extract_profiles [run] [model]
# =================================================================

# TODO:
#   - check for existing tempfiles, only load missing


def main(model, run, multithread=True, reload=False, save=True,
         config='default'):
    sim = simulation.Simulation(run=run, model=model, config=config, load_all=False)
    conf = sim.config['profile']
    params = conf['params'] + conf['composition']
    derived_params = conf['derived_params']

    if multithread:
        args = []
        for chk in sim.chk_table.index:
            args.append((chk, model, run, reload, save, params, derived_params))

        with mp.Pool(processes=4) as pool:
            pool.starmap(extract_profiles, args)
    else:
        for chk in sim.chk_table.index:
            extract_profiles(chk, model=model, run=run, reload=reload,
                             save=save, params=params, derived_params=derived_params)

    # =========================
    # bool_map = {'True': True, 'False': False}
    # for k in kwargs:
    #     kwargs[k] = bool_map[kwargs[k]]
    # batches = np.arange(batch_first, batch_last + 1)
    # burst_pipeline.run_analysis(batches, source, **kwargs)
    # =========================


def extract_profiles(chk, model, run, reload, save, params, derived_params):
    load_save.get_profile(chk, model=model, run=run, reload=reload, save=save,
                          params=params, derived_params=derived_params)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Parameters:'
              + '\n1. model'
              + '\n2. run'
              )
        sys.exit(0)
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], **dict(arg.split('=') for arg in sys.argv[3:]))
