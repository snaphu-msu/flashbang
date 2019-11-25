import sys
import multiprocessing as mp

from flashbang import simulation, load_save

# =================================================================
# Script callable from terminal to extract model profiles
# Usage:
#   python extract_profiles [run] [model]
# =================================================================


def main(run, model, multithread=True, reload=False, save=True,
         config='default'):
    sim = simulation.Simulation(run=run, model=model, config=config,
                                load_all=False)
    conf = sim.config['profile']
    params = conf['params'] + conf['composition']

    if multithread:
        args = []
        for chk_i in sim.chk_list:
            args.append((run, chk_i, model, reload, save, params))

        with mp.Pool(processes=4) as pool:
            pool.starmap(extract_profiles, args)
    else:
        for chk_i in sim.chk_list:
            extract_profiles(run, chk_i, model=model, reload=reload,
                             save=save, params=params)

    # =========================
    # bool_map = {'True': True, 'False': False}
    # for k in kwargs:
    #     kwargs[k] = bool_map[kwargs[k]]
    # batches = np.arange(batch_first, batch_last + 1)
    # burst_pipeline.run_analysis(batches, source, **kwargs)
    # =========================


def extract_profiles(run, chk_i, model, reload, save, params):
    load_save.get_profile(run, chk_i, model, reload=reload, save=save,
                          params=params)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Parameters:'
              + '\n1. run'
              + '\n2. model'
              )
        sys.exit(0)
    if len(sys.argv) == 2:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        main(sys.argv[1], sys.argv[2],
                         **dict(arg.split('=') for arg in sys.argv[3:]))
