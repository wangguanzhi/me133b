import numpy as np
import cv2
from continuous_particle_utilities import Visualization, Robot


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
    weights = 1.0 / (1.0 + l2_distance)

    postWeights = priorWeights * weights

    # Normalize.
    s = np.sum(postWeights)
    reset = False
    if s == 0.0:
        print("LOST ALL BELIEF.  STARTING OVER!!!!")
        for particle in particles:
            particle.Reset()
        postWeights = np.ones(len(particles)) / len(particles)
        s = np.sum(postWeights)
        reset = True

    postWeights = (1.0 / s) * postWeights
    return postWeights


def resample(particles, weights, numParticles):
    # Resample the particles based on the weights.
    new_particles = []
    indices = np.random.choice(len(weights), numParticles, p=weights)
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
    num_rays = 360
    lidar_range = 100
    num_particles = 100

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
        a = time.time()
        visual.Show(robot=robot, particles=particles)
        b = time.time()
        print(f"Time to show: {b - a:.3f} seconds.")

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
        a = time.time()
        computePrediction(particles, forward, turn)
        b = time.time()
        print(f"Time to computePrediction: {b - a:.3f} seconds.")

        # Correct the prediction/execute the measurement update.
        robot_sensor_reading = robot.Sensor()
        a = time.time()
        weights = updateBelief(weights, particles, robot_sensor_reading)
        b = time.time()
        print(f"Time to updateBelief: {b - a:.3f} seconds.")
        # Resample the particles.
        if 1.0 / np.sum(np.square(weights)) < len(particles) / 4.0:
            a = time.time()
            particles = resample(particles, weights, num_particles)
            b = time.time()
            print(f"Time to resample: {b - a:.3f} seconds.")
            weights = np.ones(len(particles)) / len(particles)


if __name__ == "__main__":
    main()
