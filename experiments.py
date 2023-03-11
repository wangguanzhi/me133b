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

    np.save(save_path, res_all)


def test_PF(n_runs, ns_particles, resampling_constants, save_path):

    ## Define arguments
    dist_converge_threshold = 2
    n_steps_kidnap = 5
    probCmd = 0.8
    probProximal = [0.9, 0.6, 0.3]
    visual_on = False
    verbose = False
    max_iter = 1000

    ## Run experiment with multiprocessing
    res_all = np.zeros((n_runs, len(ns_particles), len(resampling_constants), 4))

    for a, n_particles in enumerate(ns_particles):
        for b, resampling_constant in enumerate(resampling_constants):

            print("n_particles = ", n_particles, " resampling_constant = ", resampling_constant)

            ## Format arguments
            arguments = []

            for i in range(n_runs):
                arguments.append((
                    n_particles,
                    resampling_constant,
                    dist_converge_threshold,
                    n_steps_kidnap,
                    probCmd,
                    probProximal,
                    visual_on,
                    verbose,
                    max_iter))

            ## Run experiment with multiprocessing
            results = multiprocessing_wrapper(PF.run_experiment, arguments)

            ## Process results
            for i, result in enumerate(results):
                res_all[i, a, b, :] = np.array(result)

    np.save(save_path, res_all)


def test_AMCL(n_runs, 
              ns_particles, 
              resample_threshold_factors, 
              n_particles_factors, 
              aveWeights_factors, 
              save_path):

    ## Define arguments
    dist_converge_threshold = 2
    n_steps_kidnap = 5
    probCmd = 0.8
    probProximal = [0.9, 0.6, 0.3]
    visual_on = False
    verbose = False
    max_iter = 1000

    ## Run experiment with multiprocessing
    res_all = np.zeros((n_runs, 
                        len(ns_particles), 
                        len(resample_threshold_factors),
                        len(n_particles_factors),
                        len(aveWeights_factors), 5))

    for a, n_particles in enumerate(ns_particles):
        for b, resample_threshold_factor in enumerate(resample_threshold_factors):
            for c, n_particles_factor in enumerate(n_particles_factors):
                for d, aveWeights_factor in enumerate(aveWeights_factors):
                    
                    print("n_particles = ", n_particles, 
                          " resample_threshold_factor = ", resample_threshold_factor,
                          " n_particles_factor = ", n_particles_factor,
                          " aveWeights_factor = ", aveWeights_factor)

                    ## Format arguments
                    arguments = []

                    for i in range(n_runs):
                        arguments.append((
                            n_particles,
                            resample_threshold_factor,
                            n_particles_factor,
                            aveWeights_factor,
                            dist_converge_threshold,
                            n_steps_kidnap,
                            probCmd,
                            probProximal,
                            visual_on,
                            verbose,
                            max_iter))

                    ## Run experiment with multiprocessing
                    results = multiprocessing_wrapper(AMCL.run_experiment, arguments)

                    ## Process results
                    for i, result in enumerate(results):
                        res_all[i, a, b, c, d, :] = np.array(result)

    np.save(save_path, res_all)


def test_CPF(n_runs, 
             ns_particles,
             lidar_ranges,
             ns_rays,
             resampling_constants,
             sensor_diff_powers,
             reset_belief_thresholds,
             save_path):

    ## Define arguments
    dist_positon_threshold = 100
    dist_angle_threshold = 5
    n_steps_kidnap = 5
    cmd_noise = 0.1
    sensor_noise = 0.0
    visual_on = False
    verbose = False
    max_iter = 1000

    ## Run experiment with multiprocessing
    res_all = np.zeros((n_runs, 
                        len(ns_particles), 
                        len(lidar_ranges),
                        len(ns_rays),
                        len(resampling_constants),
                        len(sensor_diff_powers),
                        len(reset_belief_thresholds), 4))

    for a, n_particles in enumerate(ns_particles):
        for b, lidar_range in enumerate(lidar_ranges):
            for c, n_rays in enumerate(ns_rays):
                for d, resampling_constant in enumerate(resampling_constants):
                    for e, sensor_diff_power in enumerate(sensor_diff_powers):
                        for f, reset_belief_threshold in enumerate(reset_belief_thresholds):
                    
                            print("n_particles = ", n_particles, 
                                  "lidar_range = ", lidar_range,
                                  "n_rays = ", n_rays,
                                  " resampling_constant = ", resampling_constant,
                                  " sensor_diff_power = ", sensor_diff_power,
                                  " reset_belief_threshold = ", reset_belief_threshold)

                            ## Format arguments
                            arguments = []

                            for i in range(n_runs):
                                arguments.append((
                                    n_particles,
                                    lidar_range,
                                    n_rays,
                                    resampling_constant,
                                    sensor_diff_power,
                                    reset_belief_threshold,
                                    dist_positon_threshold,
                                    dist_angle_threshold,
                                    n_steps_kidnap,
                                    cmd_noise,
                                    sensor_noise,
                                    visual_on,
                                    verbose,
                                    max_iter))

                            ## Run experiment with multiprocessing
                            results = multiprocessing_wrapper(CPF.run_experiment, arguments)

                            ## Process results
                            for i, result in enumerate(results):
                                res_all[i, a, b, c, d, e, f, :] = np.array(result)

    np.save(save_path, res_all)


def test_CAMCL():
    pass


if __name__ == '__main__':



    ## Test Kalman filter
    # n_runs = 1000
    # save_path = 'kalman_n' + str(n_runs) + '.npy'
    # test_kalman(n_runs, save_path)

    ## Test Particle filter
    # n_runs = 1000
    # ns_particles = [10, 50, 100, 500, 1000, 5000]
    # resampling_constants = [2, 5, 10, 20, 50]
    # save_path = 'PF_n' + str(n_runs) + '.npy'
    # test_PF(n_runs, ns_particles, resampling_constants, save_path)

    ## Test Adaptive Monte Carlo
    # n_runs = 1000
    # ns_particles = [10, 100, 1000]
    # resample_threshold_factors = [1.2, 1.5, 2, 2.5]
    # n_particles_factors = [1, 2.5, 5]
    # aveWeights_factors = [1, 1.5, 2, 3] 
    # save_path = 'AMCL_n' + str(n_runs) + '.npy'
    # test_AMCL(n_runs, 
    #           ns_particles, 
    #           resample_threshold_factors, 
    #           n_particles_factors, 
    #           aveWeights_factors, 
    #           save_path)
    
    ## Test continuous Particle filter
    n_runs = 100
    # ns_particles = [100, 500, 1000, 5000]
    # lidar_ranges = [50, 150, 300]
    # ns_rays = [8, 12, 20]
    resampling_constants = [5, 10, 20, 40]
    sensor_diff_powers = [1, 1.5, 2]
    reset_belief_thresholds = [1e-6, 1e-8, 1e-10]

    ns_particles = [1000]
    lidar_ranges = [150]
    ns_rays = [8]
    # resampling_constants = [10]
    # sensor_diff_powers = [2]
    # reset_belief_thresholds = [1e-8]

    save_path = 'CPF_n' + str(n_runs) + '_1.npy'
    test_CPF(n_runs, 
             ns_particles,
             lidar_ranges,
             ns_rays,
             resampling_constants, 
             sensor_diff_powers, 
             reset_belief_thresholds, 
             save_path)
