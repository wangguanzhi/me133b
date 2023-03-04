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

from kalman_utilities import Visualization, Robot


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
#    bel         Grid of probabilities (current belief)
#    drow, dcol  Delta in row/col
#    probCmd     Modeled probability of command executing
#    prd         Grid of probabilities (prediction)
#
def computePrediction(bel, drow, dcol, probCmd=1):
    # Prepare an empty prediction grid.
    prd = np.zeros((rows, cols))

    # Iterate over/determine the probability for all (non-wall) elements.
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            # Try to shift by the given delta.
            if not walls[row + drow, col + dcol]:
                prd[row + drow, col + dcol] += probCmd * bel[row, col]
                prd[row, col] += (1.0 - probCmd) * bel[row, col]
            else:
                prd[row, col] += bel[row, col]

    # Return the prediction grid
    return prd


#
#  Measurement Update (Correction)
#
#    prior       Grid of prior probabilities (belief)
#    probSensor  Grid of probability that (sensor==True)
#    sensor      Value of sensor
#    post        Grid of posterior probabilities (updated belief)
#
def updateBelief(prior, probSensor, sensor):
    # Update the belief based on the sensor reading, which can be one
    # of two cases: (sensor==True) or (sensor==False)
    if sensor:
        post = prior * probSensor
    else:
        post = prior * (1.0 - probSensor)

    # Normalize.
    s = np.sum(post)
    reset = False
    if s == 0.0:
        post = 1.0 - walls
        s = np.sum(post)
        reset = True

    post = (1.0 / s) * post
    return post, reset


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


def run_experiment(
    dist_converge_threshold=2,
    n_steps_kidnap=5,
    probCmd=0.8,
    probProximal=[0.9, 0.6, 0.3],
    visual_on=True,
    verbose=True,
    max_iter=1000,
):

    if visual_on:
        visual = Visualization(walls)

    robot = Robot(walls, probCmd=probCmd, probProximal=probProximal, verbose=verbose)

    # Pre-compute the probability grids for each sensor reading.
    probUp = precomputeSensorProbability(-1, 0, probProximal)
    probRight = precomputeSensorProbability(0, 1, probProximal)
    probDown = precomputeSensorProbability(1, 0, probProximal)
    probLeft = precomputeSensorProbability(0, -1, probProximal)

    # Start with a uniform belief grid.
    bel = 1.0 - walls
    bel = (1.0 / np.sum(bel)) * bel

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
        if visual_on:
            visual.Show(bel, robot.Position())

        ## Check convergence
        max_bel = np.max(bel)
        max_bel_pos = np.unravel_index(np.argmax(bel, axis=None), bel.shape)

        ## L1 distance between actual robot pos and highest confidence position
        dist = np.sum(np.abs(max_bel_pos - np.array(robot.Position())))

        if verbose:
            print(
                "max belief is ",
                max_bel,
                " at ",
                max_bel_pos,
                "; distance from actual pos =  ",
                dist,
                "step_count_converge = ",
                step_count_converge,
                "step_count_reset_belief = ",
                step_count_reset_belief,
                "step_count_reconverge = ",
                step_count_reconverge,
            )

        if max_bel > 0.5 and dist < dist_converge_threshold:

            if converged and step_count_reconverge > 0:
                break

            if verbose:
                print("Converged after ", step_count_converge, " steps")

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
        prd = computePrediction(bel, drow, dcol, probCmd)
        # visual.Show(prd)
        # input("Showing the prediction")

        # Check the prediction.
        if abs(np.sum(prd) - 1.0) > 1e-12:
            print("WARNING: Prediction does not add up to 100%")

        # Correct the prediction/execute the measurement update.
        bel = prd
        bel, reset1 = updateBelief(bel, probUp, robot.Sensor(-1, 0))
        bel, reset2 = updateBelief(bel, probRight, robot.Sensor(0, 1))
        bel, reset3 = updateBelief(bel, probDown, robot.Sensor(1, 0))
        bel, reset4 = updateBelief(bel, probLeft, robot.Sensor(0, -1))

        if np.any([reset1, reset2, reset3, reset4]):
            if verbose:
                print("LOST ALL BELIEF.  STARTING OVER!!!!")
            if converged:
                belief_reset = True

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
        "[Particle Filter] step_count_converge = ",
        step_count_converge,
        " step_count_reset_belief = ",
        step_count_reset_belief,
        " step_count_reconverge = ",
        step_count_reconverge,
    )

    return step_count_converge, step_count_reset_belief, step_count_reconverge


#
#
#  Main Code
#
def main():
    # Initialize the figure.
    visual = Visualization(walls)

    # TODO... PICK WHAT THE "REALITY" SHOULD SIMULATE:
    # Initialize the robot simulation.
    # part (a)
    # robot  = Robot(walls)
    # Part (b)
    # robot = Robot(walls, row=12, col=26)
    # Part (c)
    # robot = Robot(walls, row=12, col=26, probProximal = [0.9, 0.6, 0.3])
    # Part (d), (e)
    robot = Robot(walls, row=15, col=47, probCmd=0.8, probProximal=[0.9, 0.6, 0.3])
    # And to play:
    # robot = Robot(walls, probCmd = 0.8, probProximal = [0.9, 0.6, 0.3])

    # TODO... PICK WHAT YOUR LOCALIZATION SHOULD ASSUME:
    # Pick the algorithm assumptions:
    # probCmd      = 1.0                  # Part (a/b), (c), (d)
    probCmd = 0.8  # Part (e), or to play
    # probProximal = [1.0]                # Part (a/b)
    probProximal = [0.9, 0.6, 0.3]  # Part (c), (d), (e), or to play

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

    # Show the sensor probability maps.
    visual.Show(probUp)
    input("Probability of proximal sensor up reporting True")
    visual.Show(probRight)
    input("Probability of proximal sensor right reporting True")
    visual.Show(probDown)
    input("Probability of proximal sensor down reporting True")
    visual.Show(probLeft)
    input("Probability of proximal sensor left reporting True")

    # Start with a uniform belief grid.
    bel = 1.0 - walls
    bel = (1.0 / np.sum(bel)) * bel

    # Loop continually.
    while True:
        # Show the current belief.  Also show the actual position.
        visual.Show(bel, robot.Position())

        # Show where the robot's highest belief and the position corresponding to the highest belief
        max_bel = np.max(bel)
        max_bel_pos = np.unravel_index(np.argmax(bel, axis=None), bel.shape)
        print("max belief is ", max_bel, " at ", max_bel_pos)

        # Check if the robot correctly localizes
        if max_bel_pos[0] == robot.row and max_bel_pos[1] == robot.col:
            print("Localized with confidence: ", max_bel)

        # Get the command key to determine the direction.
        while True:
            key = input("Cmd (q=quit, w=up, s=down, a=left, d=right) ?")
            if key == "q":
                return
            elif key == "w":
                (drow, dcol) = (-1, 0)
                break
            elif key == "s":
                (drow, dcol) = (1, 0)
                break
            elif key == "a":
                (drow, dcol) = (0, -1)
                break
            elif key == "d":
                (drow, dcol) = (0, 1)
                break
            elif key == "k":  ## k for kidnap
                key2 = input("Cmd enter new position of robot in the form: y, x ")
                new_pos = key2.split(", ")
                robot.row = int(new_pos[0])
                robot.col = int(new_pos[1])
                break

        # Move the robot in the simulation.
        robot.Command(drow, dcol)

        # Compute a prediction.
        prd = computePrediction(bel, drow, dcol, probCmd)
        # visual.Show(prd)
        # input("Showing the prediction")

        # Check the prediction.
        if abs(np.sum(prd) - 1.0) > 1e-12:
            print("WARNING: Prediction does not add up to 100%")

        # Correct the prediction/execute the measurement update.
        bel = prd
        bel, _ = updateBelief(bel, probUp, robot.Sensor(-1, 0))
        bel, _ = updateBelief(bel, probRight, robot.Sensor(0, 1))
        bel, _ = updateBelief(bel, probDown, robot.Sensor(1, 0))
        bel, _ = updateBelief(bel, probLeft, robot.Sensor(0, -1))


if __name__ == "__main__":

    ## Standard implementation
    # main()

    ## Run this to see single trial experiment
    run_experiment()

    # Mass experiment
    n_runs = 1000

    res_all = np.zeros((n_runs, 3))

    for i in range(n_runs):

        print("run = ", i)
        (
            step_count_converge,
            step_count_reset_belief,
            step_count_reconverge,
        ) = run_experiment(visual_on=False, verbose=False)

        res_all[i, 0] = step_count_converge
        res_all[i, 1] = step_count_reset_belief
        res_all[i, 2] = step_count_reconverge

    np.save("KF_n" + str(n_runs) + ".npy", res_all)
