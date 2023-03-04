#!/usr/bin/env python3
#
#   hw5_localize.py
#
#   Homework 5 code framework to localize a robot in a grid...
#
#   Places to edit are marked as TODO.
#
import time
import numpy as np

from particle_utilities import Visualization, Robot

np.random.seed(42)

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



def run_experiment(numParticles=1000, dist_converge_threshold=2, n_steps_kidnap=5, 
                   probCmd=0.8, probProximal=[0.9, 0.6, 0.3], visual_on=True, verbose=True, max_iter=1000):

    if visual_on:
        visual = Visualization(walls)

    robot = Robot(walls, probCmd=probCmd, probProximal=probProximal, verbose=verbose)

    # Pre-compute the probability grids for each sensor reading.
    probUp = precomputeSensorProbability(-1, 0, probProximal)
    probRight = precomputeSensorProbability(0, 1, probProximal)
    probDown = precomputeSensorProbability(1, 0, probProximal)
    probLeft = precomputeSensorProbability(0, -1, probProximal)

    # Start with a uniform belief over all particles.
    particles = [
        Robot(walls=walls, probCmd=probCmd, probProximal=probProximal)
        for _ in range(numParticles)
    ]

    weights = np.ones(numParticles) / numParticles

    # The performance variables 
    step_count_converge = 0
    step_count_reconverge = -n_steps_kidnap
    step_count_reset_belief = -n_steps_kidnap
    converged = False
    belief_reset = False
    

    # Loop continually.
    n_iter = 0
    while n_iter < max_iter:
        
        if not converged or (step_count_reconverge >= 0):
            n_iter += 1
        
        # Show the current belief.  Also show the actual position.
        bel = np.zeros((rows, cols))

        for i, particle in enumerate(particles):
            bel[particle.row, particle.col] += weights[i]
            # bel[particle.row, particle.col] += 1 / numParticles

        if visual_on:
            visual.Show(bel, robot.Position())

        ## Check convergence
        max_bel = np.max(bel)
        max_bel_pos = np.unravel_index(np.argmax(bel, axis=None), bel.shape)

        ## L1 distance between actual robot pos and highest confidence position
        dist = np.sum(np.abs(max_bel_pos - np.array(robot.Position()))) 

        if verbose:
            print('max belief is ', max_bel, 
                  ' at ', max_bel_pos, 
                  '; distance from actual pos =  ', dist,
                  'step_count_converge = ', step_count_converge,
                  'step_count_reset_belief = ', step_count_reset_belief,
                  'step_count_reconverge = ', step_count_reconverge)
            
        if max_bel > 0.5 and dist < dist_converge_threshold:

            if converged and step_count_reconverge > 0:
                break

            if verbose:
                print('Converged after ', step_count_converge, ' steps')

            converged = True


        ## Automatic random movement
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        idx = np.random.choice(np.arange(4))
        (drow, dcol) = movements[idx]

        # Move the robot in the simulation.
        robot.Command(drow, dcol)
        if visual_on:
            time.sleep(0.1)

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
                if verbose:
                    print("LOST ALL BELIEF.  STARTING OVER!!!!")
                if converged:
                    belief_reset = True
                break

        # Resample the particles.
        if 1.0 / np.sum(np.square(weights)) < len(particles) / 2.0:
            # if verbose:
            #     print('Resampling')
            particles = resample(particles, weights, numParticles)
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
                print('Kidnapping')
            robot.Reset()


    print('[Particle Filter] step_count_converge = ', step_count_converge, 
          ' step_count_reset_belief = ', step_count_reset_belief, 
          ' step_count_reconverge = ', step_count_reconverge)

    return step_count_converge, step_count_reset_belief, step_count_reconverge



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
        bel = np.zeros((rows, cols))

        for i, particle in enumerate(particles):
            bel[particle.row, particle.col] += weights[i]
            # bel[particle.row, particle.col] += 1 / numParticles

        visual.Show(bel, robot.Position())



        ## Check convergence
        max_bel = np.max(bel)
        max_bel_pos = np.unravel_index(np.argmax(bel, axis=None), bel.shape)

        ## L1 distance between actual robot pos and highest confidence position
        dist = np.sum(np.abs(max_bel_pos - np.array(robot.Position()))) 

        print('max belief is ', max_bel, ' at ', max_bel_pos, '; distance from actual pos: ', dist)
        if dist < 2:
            print('Converged')


        ## Automatic random movement
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        idx = np.random.choice(np.arange(4))
        (drow, dcol) = movements[idx]
        
        ## Get the command key to determine the direction.
        # while True:

        #     key = input("Cmd (q=quit, w=up, s=down, a=left, d=right) ?")
        #     if key == "q":
        #         return
        #     elif key == "w":
        #         (drow, dcol) = (-1, 0)
        #         break
        #     elif key == "s":
        #         (drow, dcol) = (1, 0)
        #         break
        #     elif key == "a":
        #         (drow, dcol) = (0, -1)
        #         break
        #     elif key == "d":
        #         (drow, dcol) = (0, 1)
        #         break
        #     elif key == 'k':  ## k for kidnap
        #         key2 = input("Cmd enter new position of robot in the form: y, x ")
        #         new_pos = key2.split(', ')
        #         robot.row = int(new_pos[0])
        #         robot.col = int(new_pos[1])
        #         break

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
                print("LOST ALL BELIEF.  STARTING OVER!!!!")
                break

        # Resample the particles.
        if 1.0 / np.sum(np.square(weights)) < len(particles) / 2.0:
            print('Resampling')
            particles = resample(particles, weights, numParticles)
            weights = np.ones(len(particles)) / len(particles)


if __name__ == "__main__":

    ## Standard implementation
    # main()

    ## Run this to see single trial experiment
    run_experiment()

    ## Mass experiment
    # n_runs = 1000
    # ns_particles = [10, 50, 100, 500, 1000, 5000]

    # res_all = np.zeros((n_runs, len(ns_particles), 3))

    # for i in range(n_runs):
    #     for j, n_particles in enumerate(ns_particles):

    #         print('run = ', i, ' n_particles = ', n_particles)
    #         step_count_converge, step_count_reset_belief, step_count_reconverge = run_experiment(numParticles=n_particles, 
    #                                                                                              visual_on=False, 
    #                                                                                              verbose=False)

    #         res_all[i, j, 0] = step_count_converge
    #         res_all[i, j, 1] = step_count_reset_belief
    #         res_all[i, j, 2] = step_count_reconverge

    # np.save('PF_n' + str(n_runs) + '.npy', res_all)
