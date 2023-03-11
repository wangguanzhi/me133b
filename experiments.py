import tqdm
import numpy as np
import multiprocessing.pool as mpp

import AMCL
import kalman
import particle as PF
import continuous_particle as CPF


multiprocessing_pool_count = 24


## istarmap function comes from:
##   https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap


## Code for using tqdm with istarmap comes from:
##   https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
def multiprocessing_wrapper(func, arguments):

    results = []

    with mpp.Pool(multiprocessing_pool_count) as pool:
        for result in tqdm.tqdm(pool.istarmap(func, arguments), total=len(arguments)):
            results.append(result)

    return results







def test_kalman(n_runs, save_path):

    ### Non Multiprocessing implementation 

    # res_all = np.zeros((n_runs, 4))

    # for i in range(n_runs):

    #     print("[Kalman] run = ", i)

    #     (
    #         step_count_converge,
    #         step_count_reset_belief,
    #         step_count_reconverge,
    #         runtime
    #     ) = kalman.run_experiment(visual_on=False, verbose=False)

    #     res_all[i, 0] = step_count_converge
    #     res_all[i, 1] = step_count_reset_belief
    #     res_all[i, 2] = step_count_reconverge
    #     res_all[i, 3] = runtime

    # print(res_all)

    # print(np.mean(res_all[:, 3]))
    # np.save(save_path, res_all)



    
    ### Multiprocessing implementation

    ## Define arguments
    dist_converge_threshold = 2
    n_steps_kidnap = 5
    probCmd = 0.8
    probProximal = [0.9, 0.6, 0.3]
    visual_on = False
    verbose = False
    max_iter = 1000

    ## Format arguments
    arguments = [(dist_converge_threshold, 
                  n_steps_kidnap, 
                  probCmd, 
                  probProximal, 
                  visual_on, 
                  verbose, 
                  max_iter) for i in range(n_runs)]

    ## Run experiment with multiprocessing
    results = multiprocessing_wrapper(kalman.run_experiment, arguments)

    ## Process results
    res_all = np.zeros((n_runs, 4))
    for i, result in enumerate(results):
        res_all[i, :] = np.array(result)

    print(np.mean(res_all[:, 3]))
    # np.save(save_path, res_all)



def test_PF():

    pass


def test_AMCL():

    pass


def test_CPF():

    pass


def test_CAMCL():
    pass













if __name__ == '__main__':



    ## Test Kalman filter
    n_runs = 100
    save_path = 'kalman_n' + str(n_runs) + '.npy'
    test_kalman(n_runs, save_path)


    

    

