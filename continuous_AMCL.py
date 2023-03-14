import time
import numpy as np
import cv2
import scipy.stats
from continuous_AMCL_utilities import Visualization, Robot


#
#  Define the Walls
#
w = [
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "x               x               x               x",
    "x                x             x                x",
    "x                 x           x                 x",
    "x        xxxx      x         x                  x",
    "x        x   x      x       x                   x",
    "x        x    x      x     x      xxxxx         x",
    "x        x     x      x   x     xxx   xxx       x",
    "x        x      x      x x     xx       xx      x",
    "x        x       x      x      x         x      x",
    "x        x        x           xx         xx     x",
    "x        x        x           x           x     x",
    "x        x        x           x           x     x",
    "x        x        x           x           x     x",
    "x                 xx         xx           x     x",
    "x                  x         x                  x",
    "x                  xx       xx                  x",
    "x                   xxx   xxx                   x",
    "x                     xxxxx         x           x",
    "x                                   x          xx",
    "x                                   x         xxx",
    "x            x                      x        xxxx",
    "x           xxx                     x       xxxxx",
    "x          xxxxx                    x      xxxxxx",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]
scale = 25
walls = np.array([[1.0 * (c == "x") for c in s] for s in w]).astype(np.uint8)
width = int(walls.shape[1] * scale)
height = int(walls.shape[0] * scale)
walls = cv2.resize(walls, (width, height), interpolation=cv2.INTER_NEAREST)
print(f"map height: {walls.shape[0]}, map width: {walls.shape[1]}")

def findnumOccupied(particles):  # find number of occupied grid cells
    tmp = []
    for particle in particles:
        tmp.append((int(particle.x/10), int(particle.y/10)))
    numOccupied = len(np.unique(tmp))
    return numOccupied

def score_resample(
    particles,
    weights,
    n_particles,
    cmd_noise,
    n_rays,
    lidar_range,
    walls,
    aveWeights,
    aveWeights_factor,
    prev_aveWeights,
    score_coef
):  # sample based on difference of performance
    new_particles = []
    new_weights = []
    for i in range(
        # this determines number of new particles created, can be tuned
        int(n_particles / score_coef * (prev_aveWeights - aveWeights) / prev_aveWeights)
    ):  # num of particles based on difference between previous weights and current weights
        new_particles.append(
            Robot(
                walls,
                x=0,
                y=0,
                heading=0,
                cmd_noise=cmd_noise,
                sensor_noise=None,
                n_rays=n_rays,
                lidar_range=lidar_range,
                verbose=False,
            )
        )  # random sample
        new_weights.append(
            aveWeights / aveWeights_factor + 0.00000000001
        )  # weight is based on average weight

    n = 0
    for particle in particles:
        new_particles.append(particle.Copy())
        new_weights.append(weights[n])
        n = n + 1

    new_weights = np.array(new_weights)
    return new_particles, new_weights


def KLD_resample(particles, weights, n_particles):  # resample based on KL distance
    new_particles = []
    new_weights = []
    indices = np.random.choice(len(weights), n_particles, p=weights)
    for index in indices:
        new_particles.append(particles[index].Copy())
        new_weights.append(weights[index])
    new_weights = np.array(new_weights)
    return new_particles, new_weights


## compute number of particles needed based on Kullback-Leibler distance
def computeN(particles, verbose=True):  
    k = findnumOccupied(particles)

    if verbose:
        print("numOccupied = " + str(k))

    if k == 0:
        if verbose:
            print("numOccupied == 0")
        return len(particles)
    
    quantile = 0.99  # chi-distribution quantile
    epsilon = 0.02  # K-L distance difference bound

    if k == 1:
        n = 1

    n = (
        (k - 1)
        / (2 * epsilon)
        * (
            1
            - 2 / (9 * (k - 1))
            + np.sqrt(2 / (9 * (k - 1))) * scipy.stats.norm.ppf(quantile)
        )
        ** 3
    )
    n = int(np.ceil(n)) + 1
    return n


def computePrediction(particles, forward, turn):
    for particle in particles:
        particle.Command(forward, turn)


def updateBelief(priorWeights, particles, robot_sensor_reading, sensor_diff_power=1, verbose=True):

    particle_sensor_readings = []
    for particle in particles:
        particle_sensor_reading = particle.Sensor()
        particle_sensor_readings.append(particle_sensor_reading)
    particle_sensor_readings = np.array(particle_sensor_readings)
    l2_distance = np.linalg.norm(
        np.array(robot_sensor_reading) - particle_sensor_readings, axis=1
    )
    weights = 1.0 / (1.0 + np.power(l2_distance, sensor_diff_power))

    postWeights = priorWeights * weights

    # reset = False
    # # # Normalize.
    # median_weight = np.median(postWeights)
    # if median_weight < reset_belief_threshold:

    #     reset = True
    #     if verbose:
    #         print("Resetting belief for bad particles.")

    #     for i, particle in enumerate(particles):
    #         if postWeights[i] < reset_belief_threshold:
    #             particle.Reset()
    #             postWeights[i] = median_weight

    # s = np.sum(postWeights)
    # postWeights = (1.0 / s) * postWeights
    return postWeights


def resample(particles, weights, n_particles):
    # Resample the particles based on the weights.
    new_particles = []
    indices = np.random.choice(len(weights), n_particles, p=weights)
    for index in indices:
        new_particles.append(particles[index].Copy())
    return new_particles


def run_experiment(
    n_particles=1000,
    lidar_range = 150,
    n_rays=8,
    sensor_diff_power=1,
    weight_coef=1.5,
    aveWeights_factor=1.5,
    score_coef=1,
    dist_positon_threshold=50,
    dist_angle_threshold=5,
    n_steps_kidnap=5,
    cmd_noise = 0.1,
    sensor_noise = 0.0,
    visual_on=True,
    verbose=True,
    max_iter=500,
):
    
    time_start = time.time()

    # Initialize the figure.
    if visual_on:
        visual = Visualization(walls)


    aveWeights = 0
    prev_aveWeights = 0
    count = 1


    # TODO... PICK WHAT THE "REALITY" SHOULD SIMULATE:
    robot = Robot(
        walls,
        # x=600,
        # y=200,
        # heading=0,
        cmd_noise=cmd_noise,
        sensor_noise=sensor_noise,
        n_rays=n_rays,
        lidar_range=lidar_range,
        verbose=False,
    )

    # Start with a uniform belief over all particles.
    particles = [
        Robot(
            walls,
            x=0,
            y=0,
            heading=0,
            cmd_noise=cmd_noise,
            sensor_noise=None,
            n_rays=n_rays,
            lidar_range=lidar_range,
            verbose=False,
        )
        for _ in range(n_particles)
    ]
    weights = np.ones(n_particles) / n_particles

    # The performance variables
    step_count_converge = 0
    step_count_reconverge = -n_steps_kidnap
    step_count_reset_belief = -n_steps_kidnap
    converged = False
    belief_reset = False
    n_particles_list = []

    # Loop continually.
    n_iter = 0
    while n_iter < max_iter:

        if not converged or (step_count_reconverge >= 0):
            n_iter += 1

        # Show the current belief.  Also show the actual position.
        if visual_on:
            visual.Show(robot=robot, particles=particles, particle_weights=weights)

        ## Check convergence
        robot_q = robot.get_q()
        robot_position = robot_q[0:2]
        robot_heading = robot_q[2]

        n_particles_cur = len(particles)

        particle_qs = np.zeros((n_particles_cur, 3))

        for i, particle in enumerate(particles):
            particle_qs[i, :] = particle.get_q()

        particle_positions = particle_qs[:, 0:2]
        particle_headings = particle_qs[:, 2]

        ## TODO: May need to improve angle difference calculation here
        dist_position = np.sqrt(np.sum(np.square(particle_positions - robot_position), axis=1))
        dist_heading = particle_headings - robot_heading
        dist_heading = np.abs(np.degrees(np.arctan2(np.sin(dist_heading), np.cos(dist_heading))))

        perc_converge = np.sum(np.all([dist_heading < dist_angle_threshold,
                                       dist_position < dist_positon_threshold],
                                      axis=0)) / n_particles_cur


        if verbose:
            print('avg dist_position = ', np.mean(dist_position),
                  ' avg dist_heading = ', np.mean(dist_heading))


            print('percent particles converging = ', perc_converge,
                "step_count_converge = ",
                step_count_converge,
                "step_count_reset_belief = ",
                step_count_reset_belief,
                "step_count_reconverge = ",
                step_count_reconverge,
            )

        if perc_converge > 0.5:
            if converged and step_count_reconverge > 0:
                break
            if verbose:
                print('Converged after ', step_count_converge, ' steps')
            converged = True


        ## Automatic random movement
        movements = [(1, 0), (1, 0), (-1, 0), (0, -1), (0, 1)]
        idx = np.random.choice(np.arange(len(movements)))
        (forward, turn) = movements[idx]

        # Move the robot in the simulation.
        robot.Command(forward, turn)

        # Compute a prediction.
        computePrediction(particles, forward, turn)

        # Correct the prediction/execute the measurement update.
        robot_sensor_reading = robot.Sensor()
        weights = updateBelief(weights, particles, robot_sensor_reading, sensor_diff_power)

        # this prev_aveWeights update can be modified
        prev_aveWeights = (prev_aveWeights + aveWeights) / 2

        count += 1

        reset = False
        aveWeights = np.mean(weights)
        if prev_aveWeights > weight_coef * aveWeights: # weight_coef can be tuned
          # if the particles with high probability is discarded need sample more particles
            particles, weights = score_resample(
                particles,
                weights,
                n_particles,
                cmd_noise,
                n_rays,
                lidar_range,
                walls,
                aveWeights,
                aveWeights_factor,
                prev_aveWeights,
                score_coef)
            reset = True
        
        if reset and converged:    
            belief_reset = True
            
        weights = (1.0 / np.sum(weights)) * weights

        # Resample the particles.
        # numParticles = min(computeN(particles), 2000)
        numParticles = computeN(particles, verbose=verbose)

        particles, weights = KLD_resample(particles, weights, numParticles)
        weights = (1.0 / np.sum(weights)) * weights

        n_particles_list.append(n_particles)

        if converged:
            step_count_reconverge += 1
            if not belief_reset:
                step_count_reset_belief += 1
        else:
            step_count_converge += 1

        ## Kidnapping robot
        if step_count_reconverge == 0:
            if verbose:
                print("Kidnapping")
            robot.Reset()

    if verbose:
        print(
            "[Continuous Particle Filter] step_count_converge = ",
            step_count_converge,
            " step_count_reset_belief = ",
            step_count_reset_belief,
            " step_count_reconverge = ",
            step_count_reconverge,
            " runtime = ",
            time.time() - time_start
        )

    return sum(n_particles_list), step_count_converge, step_count_reset_belief, step_count_reconverge, time.time() - time_start









#
#  Main Code
#
def main():
    # Initialize the figure.
    visual = Visualization(walls)

    # TODO... PICK WHAT YOUR LOCALIZATION SHOULD ASSUME:
    # Pick the algorithm assumptions:
    cmd_noise = 0.1
    sensor_noise = 0.0
    n_rays = 8
    lidar_range = 100
    n_particles = 1000
    aveWeights = 0
    prev_aveWeights = 0
    count = 1
    weight_coef = 2
    score_coef = 1
    power_coef = 1
    # TODO... PICK WHAT THE "REALITY" SHOULD SIMULATE:
    robot = Robot(
        walls,
        x=600,
        y=200,
        heading=0,
        cmd_noise=cmd_noise,
        sensor_noise=sensor_noise,
        n_rays=n_rays,
        lidar_range=lidar_range,
        verbose=False,
    )

    # Start with a uniform belief over all particles.
    particles = [
        Robot(
            walls,
            x=0,
            y=0,
            heading=0,
            cmd_noise=cmd_noise,
            sensor_noise=None,
            n_rays=n_rays,
            lidar_range=lidar_range,
            verbose=False,
        )
        for _ in range(n_particles)
    ]
    weights = np.ones(n_particles) / n_particles

    # Loop continually.
    import time

    while True:
        # Show the current belief.  Also show the actual position.
        visual.Show(robot=robot, particles=particles)

        # Get the command key to determine the direction.
        while True:
            key = input(
                "Cmd (q=quit, w=forward, s=backward, a=turn_left, d=turn_right) ?"
            )
            if key == "q":
                return
            elif key == "w":
                (forward, turn) = (1, 0)
                break
            elif key == "s":
                (forward, turn) = (-1, 0)
                break
            elif key == "a":
                (forward, turn) = (0, 1)
                break
            elif key == "d":
                (forward, turn) = (0, -1)
                break

        # Move the robot in the simulation.
        robot.Command(forward, turn)

        # Compute a prediction.
        computePrediction(particles, forward, turn)

        # Correct the prediction/execute the measurement update.
        robot_sensor_reading = robot.Sensor()
        weights, reset = updateBelief(weights, particles, robot_sensor_reading)

        # this prev_aveWeights update can be modified
        prev_aveWeights = (prev_aveWeights + aveWeights) / 2

        count = count + 1
        aveWeights = np.mean(weights)
        if (
                prev_aveWeights > weight_coef * aveWeights # weight_coef can be tuned
        ):  # if the particles with high probability is discarded need sample more particles
            particles, weights = score_resample(
                particles,
                weights,
                n_particles,
                cmd_noise,
                n_rays,
                lidar_range,
                walls,
                aveWeights,
                prev_aveWeights,
                score_coef,
            )
        weights = (1.0 / np.sum(weights)) * weights

        # Resample the particles.
        numParticles = computeN(particles)
        particles, weights = KLD_resample(particles, weights, numParticles)
        weights = (1.0 / np.sum(weights)) * weights
        print(len(particles))


if __name__ == "__main__":
    # main()
    run_experiment()
