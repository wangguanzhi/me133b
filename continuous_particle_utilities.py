import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import random
import cv2


class Visualization:
    def __init__(self, walls):

        self.walls = walls
        self.height = walls.shape[0]
        self.width = walls.shape[1]

        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(False)
        plt.gca().axis("off")
        plt.gca().set_aspect("equal")
        plt.gca().set_xlim(0, self.width)
        plt.gca().set_ylim(self.height, 0)

        # Clear the content and mark.  Then show with zeros.
        self.content = None
        self.mark_robot = None
        self.mark_lidar = None
        self.mark_particles = None
        self.Show()

    def Flush(self):
        # Show the plot.
        plt.gca().axis("off")
        plt.pause(0.001)

    def MarkRobot(self, robot):
        # Potentially remove the previous mark.
        if self.mark_robot is not None:
            self.mark_robot.remove()
            self.mark_robot = None
        if self.mark_lidar is not None:
            self.mark_lidar.remove()
            self.mark_lidar = None

        x = robot.x
        y = robot.y
        heading = robot.heading

        # Check the x/y arguments.
        assert (x >= 0) and (x < self.width), "Illegal x"
        assert (y >= 0) and (y < self.height), "Illegal y"

        # Draw the mark.
        row = round(self.height - y)
        col = round(x)
        self.mark_robot = plt.gca().arrow(
            col,
            row,
            50 * np.cos(heading),
            -50 * np.sin(heading),
            width=10,
            length_includes_head=True,
            color="b",
        )
        self.mark_lidar = plt.gca().add_patch(
            Circle((col, row), robot.lidar_range, color="g", fill=False, linewidth=2)
        )

    def MarkParticles(self, particles):
        if self.mark_particles is not None:
            self.mark_particles.remove()
            self.mark_particles = None

        X = []
        Y = []
        U = []
        V = []
        for particle in particles:
            x = particle.x
            y = particle.y
            heading = particle.heading

            # Check the x/y arguments.
            assert (x >= 0) and (x < self.width), "Illegal x"
            assert (y >= 0) and (y < self.height), "Illegal y"

            X.append(round(x))
            Y.append(round(self.height - y))
            U.append(np.cos(heading))
            V.append(np.sin(heading))
        self.mark_particles = plt.gca().quiver(
            X, Y, U, V, scale=30, headwidth=2, headlength=5, color="r"
        )

    def Grid(self):

        # Potentially remove the previous grid/content.
        if self.content is not None:
            self.content.remove()
            self.content = None

        # Draw the map.
        self.content = plt.gca().imshow(
            np.stack([(1 - self.walls) * 255] * 3, axis=-1),
            aspect="equal",
            interpolation="none",
            extent=[0, self.width, self.height, 0],
            zorder=0,
        )

    def Show(self, robot=None, particles=None):
        # Update the content.
        self.Grid()

        # Potentially add the mark.
        if robot is not None:
            self.MarkRobot(robot)

        # Potentially add the particles.
        if particles is not None:
            self.MarkParticles(particles)

        # Flush the figure.
        self.Flush()


class Robot:
    def __init__(
        self,
        walls,
        x=0,
        y=0,
        heading=0,
        cmd_noise=0.1,
        sensor_noise=0.1,
        num_rays=360,
        lidar_range=100,
        verbose=False,
    ):
        self.walls = walls
        self.height = walls.shape[0]
        self.width = walls.shape[1]

        # Check the x/y arguments.
        assert (x >= 0) and (x < self.width), "Illegal x"
        assert (y >= 0) and (y < self.height), "Illegal y"
        col = round(x)
        row = round(self.height - y)

        if verbose:
            # Report.
            if self.walls[row, col]:
                location = " (random location)"
            else:
                location = " (at %d, %d)" % (x, y)
            print(
                "Starting robot with real cmd_noise = "
                + str(cmd_noise)
                + " and sensor_noise = "
                + str(sensor_noise)
                + location
            )

        self.x = x
        self.y = y
        self.heading = heading
        self.cmd_noise = cmd_noise
        self.sensor_noise = sensor_noise
        self.num_rays = num_rays
        self.lidar_range = lidar_range

        # Pick a valid starting location (if not already given).
        if row >= self.height or col >= self.width or self.walls[row, col]:
            while True:
                x = random.uniform(0, self.width)
                y = random.uniform(0, self.height)
                col = max(0, min(round(x), self.width - 1))
                row = max(0, min(round(self.height - y), self.height - 1))
                if not self.walls[row, col]:
                    break
            self.x = x
            self.y = y
            self.heading = random.random() * 2 * np.pi

    def Command(self, forward=0, turn=0):
        assert (abs(forward) <= 1) and (abs(turn) <= 1), "Bad command"
        if forward != 0:
            assert turn == 0, "Cannot move forward and turn"
        if turn != 0:
            assert forward == 0, "Cannot move forward and turn"
        x_target = (
            self.x
            + (forward + random.gauss(0, self.cmd_noise)) * np.cos(self.heading) * 20
        )
        y_target = (
            self.y
            + (forward + random.gauss(0, self.cmd_noise)) * np.sin(self.heading) * 20
        )
        xc = round(self.x)
        yc = round(self.y)
        x = round(x_target)
        y = round(y_target)
        if xc == x:
            slope = float("inf")
        else:
            slope = (y - yc) / (x - xc)
        hit = False
        if abs(slope) <= 1:
            if xc <= x:
                for i in range(xc, x + 1):
                    j = round(yc + slope * (i - xc))
                    col = i
                    row = self.height - j
                    if (
                        col < 0
                        or col >= self.width
                        or row < 0
                        or row >= self.height
                        or self.walls[row, col]
                    ):
                        hit = True
                        break
            else:
                for i in range(xc, x - 1, -1):
                    j = round(yc + slope * (i - xc))
                    col = i
                    row = self.height - j
                    if (
                        col < 0
                        or col >= self.width
                        or row < 0
                        or row >= self.height
                        or self.walls[row, col]
                    ):
                        hit = True
                        break
        else:
            if yc <= y:
                for j in range(yc, y + 1):
                    i = round(xc + (j - yc) / slope)
                    col = i
                    row = self.height - j
                    if (
                        col < 0
                        or col >= self.width
                        or row < 0
                        or row >= self.height
                        or self.walls[row, col]
                    ):
                        hit = True
                        break
            else:
                for j in range(yc, y - 1, -1):
                    i = round(xc + (j - yc) / slope)
                    col = i
                    row = self.height - j
                    if (
                        col < 0
                        or col >= self.width
                        or row < 0
                        or row >= self.height
                        or self.walls[row, col]
                    ):
                        hit = True
                        break
        if not hit:
            self.x = x_target
            self.y = y_target
        self.heading += (turn + random.gauss(0, self.cmd_noise)) * np.pi / 4

    def Sensor(self):
        noise = 0.0 if self.sensor_noise is None else self.sensor_noise
        distances = []
        for i in range(self.num_rays):
            angle = self.heading + i * 2 * np.pi / self.num_rays
            xc = round(self.x)
            yc = round(self.y)
            x = round(self.x + self.lidar_range * np.cos(angle))
            y = round(self.y + self.lidar_range * np.sin(angle))
            if xc == x:
                slope = float("inf")
            else:
                slope = (y - yc) / (x - xc)
            distance = 2 * self.lidar_range
            if abs(slope) <= 1:
                if xc <= x:
                    for i in range(xc, x + 1):
                        j = round(yc + slope * (i - xc))
                        col = i
                        row = self.height - j
                        if self.walls[row, col]:
                            distance = np.sqrt((xc - i) ** 2 + (yc - j) ** 2)
                            break
                else:
                    for i in range(xc, x - 1, -1):
                        j = round(yc + slope * (i - xc))
                        col = i
                        row = self.height - j
                        if self.walls[row, col]:
                            distance = np.sqrt((xc - i) ** 2 + (yc - j) ** 2)
                            break
            else:
                if yc <= y:
                    for j in range(yc, y + 1):
                        i = round(xc + (j - yc) / slope)
                        col = i
                        row = self.height - j
                        if self.walls[row, col]:
                            distance = np.sqrt((xc - i) ** 2 + (yc - j) ** 2)
                            break
                else:
                    for j in range(yc, y - 1, -1):
                        i = round(xc + (j - yc) / slope)
                        col = i
                        row = self.height - j
                        if self.walls[row, col]:
                            distance = np.sqrt((xc - i) ** 2 + (yc - j) ** 2)
                            break
            distances.append(distance * (1 + random.gauss(0, noise)))
        return distances

    def Reset(self):
        # Pick a valid starting location.
        while True:
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            col = max(0, min(round(x), self.width - 1))
            row = max(0, min(round(self.height - y), self.height - 1))
            if not self.walls[row, col]:
                break
        self.x = x
        self.y = y
        self.heading = random.random() * 2 * np.pi

    def Copy(self):
        return Robot(
            self.walls,
            self.x,
            self.y,
            self.heading,
            self.cmd_noise,
            self.sensor_noise,
            self.num_rays,
            self.lidar_range,
            verbose=False,
        )
