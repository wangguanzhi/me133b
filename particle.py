#!/usr/bin/env python3
#
#   hw5_localize.py
#
#   Homework 5 code framework to localize a robot in a grid...
#
#   Places to edit are marked as TODO.
#
import numpy as np

from particle_utilities import Visualization, Robot


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

walls = np.array([[1.0 * (c == "x") for c in s] for s in w])
rows = np.size(walls, axis=0)
cols = np.size(walls, axis=1)

#
#  Prediction
#
#    particles   List of particles
#    drow, dcol  Delta in row/col
#
def computePrediction(particles, drow, dcol):
    for particle in particles:
        particle.Command(drow, dcol)


#
#  Measurement Update (Correction)
#
#    priorWeights   Grid of prior probabilities
#    probSensor     Grid of probability that (sensor==True)
#    sensor         Value of sensor
#    postWeights    Grid of posterior probabilities (updated belief)
#
def updateBelief(priorWeights, particles, probSensor, sensor):
    # Update the belief based on the sensor reading, which can be one
    # of two cases: (sensor==True) or (sensor==False)
    probParticles = np.array(
        [probSensor[particle.row, particle.col] for particle in particles]
    )
    if sensor:
        postWeights = priorWeights * probParticles
    else:
        postWeights = priorWeights * (1.0 - probParticles)

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
    return postWeights, reset


#
#  Resampling
#
#    particles  List of particles
#    weights    List of weights
#    numParticles  Number of particles to resample
#
def resample(particles, weights, numParticles):
    # Resample the particles based on the weights.
    new_particles = []
    indices = np.random.choice(len(weights), numParticles, p=weights)
    for index in indices:
        new_particles.append(particles[index].Copy())
    return new_particles


#
#  Pre-compute the Sensor Probability Grid
#
#    drow, dcol    Direction in row/col
#    probProximal  List of probability that sensor triggers at dist=(index+1)
#    prob          Grid of probability that (sensor==True)
#
def precomputeSensorProbability(drow, dcol, probProximal=[1.0]):
    # Prepare an empty probability grid.
    prob = np.zeros((rows, cols))

    # Pre-compute the probability for each grid element, knowing the
    # walls and the direction of the sensor.
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            # Check for a wall in the given delta direction.
            for k in range(len(probProximal)):
                if walls[row + drow * (k + 1), col + dcol * (k + 1)]:
                    prob[row, col] = probProximal[k]
                    break

    # Return the computed grid.
    return prob


#
#
#  Main Code
#
def main():
    # Initialize the figure.
    visual = Visualization(walls)

    # TODO... PICK WHAT YOUR LOCALIZATION SHOULD ASSUME:
    # Pick the algorithm assumptions:
    # probCmd = 1.0
    probCmd = 0.8
    # probProximal = [1.0]
    probProximal = [0.9, 0.6, 0.3]
    numParticles = 1000

    # TODO... PICK WHAT THE "REALITY" SHOULD SIMULATE:
    robot = Robot(
        walls, row=15, col=47, probCmd=probCmd, probProximal=probProximal, verbose=True
    )

    # Report.
    print(
        "Localization is assuming probCmd = "
        + str(probCmd)
        + " and probProximal = "
        + str(probProximal)
    )

    # Pre-compute the probability grids for each sensor reading.
    probUp = precomputeSensorProbability(-1, 0, probProximal)
    probRight = precomputeSensorProbability(0, 1, probProximal)
    probDown = precomputeSensorProbability(1, 0, probProximal)
    probLeft = precomputeSensorProbability(0, -1, probProximal)

    # # Show the sensor probability maps.
    # visual.Show(probUp)
    # input("Probability of proximal sensor up reporting True")
    # visual.Show(probRight)
    # input("Probability of proximal sensor right reporting True")
    # visual.Show(probDown)
    # input("Probability of proximal sensor down reporting True")
    # visual.Show(probLeft)
    # input("Probability of proximal sensor left reporting True")

    # Start with a uniform belief over all particles.
    particles = [
        Robot(walls=walls, probCmd=probCmd, probProximal=probProximal)
        for _ in range(numParticles)
    ]
    weights = np.ones(numParticles) / numParticles

    # Loop continually.
    while True:
        # Show the current belief.  Also show the actual position.
        prob = np.zeros((rows, cols))
        for particle in particles:
            prob[particle.row, particle.col] = 1
        for particle in particles:
            prob[particle.row, particle.col] -= 0.5 / len(particles)
        visual.Show(prob, robot.Position())

        # Get the command key to determine the direction.
        while True:
            key = input("Cmd (q=quit, i=up, m=down, j=left, k=right) ?")
            if key == "q":
                return
            elif key == "i":
                (drow, dcol) = (-1, 0)
                break
            elif key == "m":
                (drow, dcol) = (1, 0)
                break
            elif key == "j":
                (drow, dcol) = (0, -1)
                break
            elif key == "k":
                (drow, dcol) = (0, 1)
                break

        # Move the robot in the simulation.
        robot.Command(drow, dcol)

        # Compute a prediction.
        computePrediction(particles, drow, dcol)

        # Correct the prediction/execute the measurement update.
        sensors = [
            robot.Sensor(-1, 0),
            robot.Sensor(0, 1),
            robot.Sensor(1, 0),
            robot.Sensor(0, -1),
        ]
        probSensors = [probUp, probRight, probDown, probLeft]
        for probSensor, sensor in zip(probSensors, sensors):
            weights, reset = updateBelief(weights, particles, probSensor, sensor)
            if reset:
                break

        # Resample the particles.
        if 1.0 / np.sum(np.square(weights)) < len(particles) / 2.0:
            particles = resample(particles, weights, numParticles)
            weights = np.ones(len(particles)) / len(particles)


if __name__ == "__main__":
    main()
