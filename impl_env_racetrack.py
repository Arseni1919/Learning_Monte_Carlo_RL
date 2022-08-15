import random

import matplotlib.pyplot as plt
import numpy as np


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


class RacetrackEnv:
    def __init__(self, side_size=30):
        self.side_size = side_size
        self.max_vel = 5
        self.n_actions = 3
        # create mat
        starting_line = list(range(int(0.33 * self.side_size)))
        finish_line = list(range(int(0.6 * self.side_size), self.side_size))
        mat = np.zeros((self.side_size, self.side_size))

        connect_x = random.choice(starting_line)
        connect_y = random.choice(finish_line)

        mat[connect_x, connect_y] = -1

        for y in range(connect_y):
            adding = random.choice([-1, 0, 1])
            mat[starting_line, y] = -1

        for x in range(connect_x, self.side_size - 1):
            adding = random.choice([-1, 0, 1])
            mat[x, finish_line] = -1

        mat[starting_line, 0] = -2
        mat[self.side_size - 1, finish_line] = 1

        self.mat = mat
        self.starting_line = starting_line
        self.finish_line = finish_line

    def reset(self):
        """
        state: x, y, vel_x, vel_y
        """
        start_state = (random.choice(self.starting_line), 0, 0, 0)
        return start_state

    def step(self, state, action):
        x, y, vel_x, vel_y = state
        acc_x, acc_y = action
        new_vel_x = int(max(1, min(4, vel_x + acc_x)))
        new_vel_y = int(max(1, min(4, vel_y + acc_y)))
        new_x = int(x + new_vel_x)
        new_y = int(y + new_vel_y)

        # case: inside square
        if 0 < new_x < self.side_size - 1:
            if 0 < new_y < self.side_size:
                on_mat = self.mat[new_x, new_y]

                if on_mat != 0.0:
                    # inside map
                    new_state = (new_x, new_y, new_vel_x, new_vel_y)
                    return new_state, -1, False

        # case: outside square
        if self.side_size - 1 <= new_x:
            if new_y in self.finish_line:
                # intersects with finish
                return None, -1, True

        new_state = self.reset()
        return new_state, -1, False

    def init_q_func(self):
        """
        state: x, y, vel_x, vel_y
        actions: ac_x, ac_y
        each action: -1, 0, +1
        """
        return np.ones((self.side_size, self.side_size, self.max_vel, self.max_vel, self.n_actions, self.n_actions)) * -1000

    def init_c_func(self):
        """
        state: x, y, vel_x, vel_y
        actions: ac_x, ac_y
        each action: -1, 0, +1
        """
        return np.zeros((self.side_size, self.side_size, self.max_vel, self.max_vel, self.n_actions, self.n_actions))

    def init_policy(self):
        return np.zeros((self.side_size, self.side_size, self.max_vel, self.max_vel, 2))

    def get_actions(self):
        return [-1, 0, 1]

    def render(self):
        plt.matshow(self.mat)
        plt.show()


def main():
    env = RacetrackEnv()
    env.render()


if __name__ == '__main__':
    main()



