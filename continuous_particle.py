import time
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


def updateBelief(priorWeights, particles, robot_sensor_reading, reset_belief_threshold=1e-6):

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
    median_weight = np.median(postWeights)
    if median_weight < reset_belief_threshold:
        print("Resetting belief for bad particles.")
        for i, particle in enumerate(particles):
            if postWeights[i] < reset_belief_threshold:
                particle.Reset()
                postWeights[i] = median_weight

    s = np.sum(postWeights)
    postWeights = (1.0 / s) * postWeights
    return postWeights


def resample(particles, weights, numParticles):
    # Resample the particles based on the weights.
    new_particles = []
    indices = np.random.choice(len(weights), numParticles, p=weights)
    for index in indices:
        new_particles.append(particles[index].Copy())
    return new_particles






def run_experiment(
    num_particles=1000,
    cmd_noise = 0.1,
    sensor_noise = 0.0,
    num_rays = 8,
    lidar_range = 300,
    dist_positon_threshold=100,
    dist_angle_threshold=5,
    n_steps_kidnap=5,
    resampling_constant=5.0,
    visual_on=True,
    verbose=True,
    max_iter=1000,
):

    if visual_on:
        visual = Visualization(walls)

    robot = Robot(
        walls,
        # x=600,
        # y=200,
        # heading=0,
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

    # The performance variables
    step_count_converge = 0
    step_count_reconverge = -n_steps_kidnap
    step_count_reset_belief = -n_steps_kidnap
    converged = False
    belief_reset = False

    # Loop continually.
    n_iter = 0
    while n_iter < max_iter:
        # Show the current belief.  Also show the actual position.

        if visual_on:
            visual.Show(robot=robot, particles=particles, particle_weights=weights)

        ## Check convergence
        robot_q = robot.get_q()
        robot_position = robot_q[0:2]
        robot_heading = np.degrees(robot_q[2]) % 360

        particle_qs = np.zeros((num_particles, 3))
        for i, particle in enumerate(particles):
            particle_qs[i, :] = particle.get_q()

        particle_positions = particle_qs[:, 0:2]
        particle_headings = np.degrees(particle_qs[:, 2]) % 360

        # particle_headings[particle_headings > 180] -= 360

        # if robot_heading > 180


        dist_position = np.sqrt(np.sum(np.square(particle_positions - robot_position), axis=1))
        dist_heading = np.abs(particle_headings - robot_heading)

        dist_heading

        perc_converge = np.sum(np.all([dist_heading < dist_angle_threshold,
                                       dist_position < dist_positon_threshold],
                                      axis=0)) / num_particles


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
        movements = [(1, 0), (1, 0), (1, 0),
                     (-1, 0),
                     (0, -1), (0, 1)]
        idx = np.random.choice(np.arange(len(movements)))
        (forward, turn) = movements[idx]

        # Get the command key to determine the direction.
        # while True:
        #     key = input("Cmd (q=quit, w=forward, s=backward, a=turn_left, d=turn_right) ?")
        #     if key == "q":
        #         return
        #     elif key == "w":
        #         (forward, turn) = (1, 0)
        #         break
        #     elif key == "s":
        #         (forward, turn) = (-1, 0)
        #         break
        #     elif key == "a":
        #         (forward, turn) = (0, 1)
        #         break
        #     elif key == "d":
        #         (forward, turn) = (0, -1)
        #         break

        # Move the robot in the simulation.
        robot.Command(forward, turn)

        # Compute a prediction.
        computePrediction(particles, forward, turn)

        # Correct the prediction/execute the measurement update.
        robot_sensor_reading = robot.Sensor()
        weights = updateBelief(weights, particles, robot_sensor_reading)

        # Resample the particles.
        if 1.0 / np.sum(np.square(weights)) < len(particles) / resampling_constant:
            particles = resample(particles, weights, num_particles)
            weights = np.ones(len(particles)) / len(particles)

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

    print(
        "[Continuous Particle Filter] step_count_converge = ",
        step_count_converge,
        " step_count_reset_belief = ",
        step_count_reset_belief,
        " step_count_reconverge = ",
        step_count_reconverge,
    )

    return step_count_converge, step_count_reset_belief, step_count_reconverge








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
    reset_belief_threshold = 1e-6

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
        weights = updateBelief(weights, particles, robot_sensor_reading, reset_belief_threshold)
        # Resample the particles.
        if 1.0 / np.sum(np.square(weights)) < len(particles) / 40.0:
            particles = resample(particles, weights, num_particles)
            weights = np.ones(len(particles)) / len(particles)


if __name__ == "__main__":
    # main()
    run_experiment(visual_on=True)
