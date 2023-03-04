#!/usr/bin/env python3
#
#   hw5_localize.py
#
#   Homework 5 code framework to localize a robot in a grid...
#
#   Places to edit are marked as TODO.
#
import numpy as np
import scipy.stats
import matplotlib as mpl
mpl.use('macosx')
from AMCL_utilities import Visualization, Robot


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
def findnumOccupied(particles): # find number of occupied grid cells
    tmp = []
    for particle in particles:
        tmp.append((particle.row,particle.col))
    numOccupied = len(np.unique(tmp))
    return numOccupied

def computeN(particles): # compute number of particles needed based on Kullback-Leibler distance
    k = findnumOccupied(particles)
    print("numOccupied = " + str(k))
    if k == 0:
        print("numOccupied == 0")
        return len(particles)
    quantile = 0.99 #chi-distribution quantile
    epsilon = 0.02 #K-L distance difference bound
    n = (k - 1) / (2 * epsilon) * (1 - 2/(9*(k-1)) + np.sqrt(2/(9*(k-1)))*scipy.stats.norm.ppf(quantile))**3
    n = int(np.ceil(n))
    return n
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
    return postWeights, reset


#
#  Resampling
#
#    particles  List of particles
#    weights    List of weights
#    numParticles  Number of particles to resample
#
def update_w(aveWeights,w_slow,w_fast,alpha_fast,alpha_slow): # not used for right now
    #aveWeights = np.mean(weights)
    w_fast = w_fast + alpha_fast * (aveWeights - w_fast) + 0.1 * alpha_fast * (aveWeights - w_fast)*(np.sign(aveWeights - w_fast)+1)
    w_slow = w_slow + alpha_slow * (aveWeights - w_slow)
    print("w_fast" + str(w_fast))
    print("w_slow" + str(w_slow))
    print("aveWeights" + str(aveWeights))
    return w_slow, w_fast

def score_resample(particles, weights, numParticles,probCmd,probProximal,walls,aveWeights,prev_aveWeights): # sample based on difference of performance
    new_particles = []
    new_weights = []
    for i in range(int(numParticles/5 * (prev_aveWeights - aveWeights)/prev_aveWeights)): # num of particles based on difference between previous weights and current weights
        new_particles.append(Robot(walls=walls, probCmd=probCmd, probProximal=probProximal)) # random sample
        new_weights.append(aveWeights/2 +0.00000000001) # weight is based on average weight

    n = 0
    for particle in particles:
        new_particles.append(particle.Copy())
        new_weights.append(weights[n])
        n = n + 1

    new_weights = np.array(new_weights)
    return new_particles, new_weights

def KLD_resample(particles, weights, numParticles): # resample based on KL distance
    new_particles = []
    new_weights = []
    indices = np.random.choice(len(weights), numParticles, p=weights)
    for index in indices:
        new_particles.append(particles[index].Copy())
        new_weights.append(weights[index])
    new_weights = np.array(new_weights)
    return new_particles , new_weights

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
    probCmd = 1.0
    #probCmd = 0.8
    # probProximal = [1.0]
    probProximal = [0.9, 0.6, 0.3]
    numParticles = 1000
    w_slow = 0
    w_fast = 0
    alpha_slow = 0.1
    alpha_fast = 0.5
    aveWeights = 0
    prev_aveWeights = 0
    count = 1
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
            elif key == "a":
                robot.Kidnap()
                print((robot.row,robot.col))
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

        prev_aveWeights = (prev_aveWeights + aveWeights) / 2
        count = count + 1
        aveWeights = np.mean(weights)
        w_slow, w_fast = update_w(aveWeights,w_slow,w_fast,alpha_fast,alpha_slow)
        if prev_aveWeights > 2 * aveWeights: #if the particles with high probability is discarded need sample more particles
            particles,weights = score_resample(particles, weights, numParticles, probCmd, probProximal, walls,aveWeights,prev_aveWeights)
        weights = (1.0 / np.sum(weights)) * weights
        numParticles = computeN(particles)
        particles,weights = KLD_resample(particles, weights, numParticles)
        weights = (1.0 / np.sum(weights)) * weights

if __name__ == "__main__":
    main()
