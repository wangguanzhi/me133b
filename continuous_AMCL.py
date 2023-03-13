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
scale = 50
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
    num_particles,
    cmd_noise,
    num_rays,
    lidar_range,
    walls,
    aveWeights,
    prev_aveWeights,
    score_coef
):  # sample based on difference of performance
    new_particles = []
    new_weights = []
    for i in range(
        # this determines number of new particles created, can be tuned
        int(num_particles / score_coef * (prev_aveWeights - aveWeights) / prev_aveWeights)
    ):  # num of particles based on difference between previous weights and current weights
        new_particles.append(
            Robot(
                walls,
                x=0,
                y=0,
                heading=0,
                cmd_noise=cmd_noise,
                sensor_noise=None,
                num_rays=num_rays,
                lidar_range=lidar_range,
                verbose=False,
            )
        )  # random sample
        new_weights.append(
            aveWeights / 2 + 0.00000000001
        )  # weight is based on average weight

    n = 0
    for particle in particles:
        new_particles.append(particle.Copy())
        new_weights.append(weights[n])
        n = n + 1

    new_weights = np.array(new_weights)
    return new_particles, new_weights


def KLD_resample(particles, weights, num_particles):  # resample based on KL distance
    new_particles = []
    new_weights = []
    indices = np.random.choice(len(weights), num_particles, p=weights)
    for index in indices:
        new_particles.append(particles[index].Copy())
        new_weights.append(weights[index])
    new_weights = np.array(new_weights)
    return new_particles, new_weights

def computeN(
    particles,
):  # compute number of particles needed based on Kullback-Leibler distance
    k = findnumOccupied(particles)
    print("numOccupied = " + str(k))
    if k == 0:
        print("numOccupied == 0")
        return len(particles)
    quantile = 0.98  # chi-distribution quantile
    epsilon = 0.02  # K-L distance difference bound
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
    n = int(np.ceil(n))
    return n

def computePrediction(particles, forward, turn):
    for particle in particles:
        particle.Command(forward, turn)


def updateBelief(priorWeights, particles, robot_sensor_reading):

    particle_sensor_readings = []
    for particle in particles:
        particle_sensor_reading = particle.Sensor()
        particle_sensor_readings.append(particle_sensor_reading)
    particle_sensor_readings = np.array(particle_sensor_readings)
    l2_distance = np.linalg.norm(
        np.array(robot_sensor_reading) - particle_sensor_readings, axis=1
    )
    weights = 1.0 / (1.0 + l2_distance ** power_coef)

    postWeights = priorWeights * weights

    # Normalize.
    #s = np.sum(postWeights)
    #reset = False
    #if s == 0.0:
    #    print("LOST ALL BELIEF.  STARTING OVER!!!!")
    #    for particle in particles:
    #        particle.Reset()
    #    postWeights = np.ones(len(particles)) / len(particles)
    #    s = np.sum(postWeights)
    #    reset = True

    #postWeights = (1.0 / s) * postWeights
    return postWeights


def resample(particles, weights, num_particles):
    # Resample the particles based on the weights.
    new_particles = []
    indices = np.random.choice(len(weights), num_particles, p=weights)
    for index in indices:
        new_particles.append(particles[index].Copy())
    return new_particles


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
    num_rays = 8
    lidar_range = 100
    num_particles = 1000
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
        num_rays=num_rays,
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
            num_rays=num_rays,
            lidar_range=lidar_range,
            verbose=False,
        )
        for _ in range(num_particles)
    ]
    weights = np.ones(num_particles) / num_particles

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
        weights = updateBelief(weights, particles, robot_sensor_reading)

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
                num_particles,
                cmd_noise,
                num_rays,
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
    main()
