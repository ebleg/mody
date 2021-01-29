import numpy as np
from numpy.linalg import norm
import parameters as par
from scipy.integrate import RK45
from scipy.stats import geom


def simulate_hybrid(f, t_bound, x0):
    # Initialize integrator
    integrator = RK45(f, t_bound[0], t_bound[1], x0)

    # Initialize Moore machine
    return None


def check_collision(qdq, pos_funs):
    # pos_fun: list of functions (of qdq) that determine the position of a ball
    # in the inertial frame

    # -- Collisions between balls
    # Construct matrix with relative distances between all the balls
    n_balls = len(pos_funs)
    dist_mat = np.empty((n_balls, n_balls), dtype="float")

    for i in range(n_balls):
        for j in range(n_balls):
            # Compute distance between each of the balls
            dist_mat[i, j] = norm(pos_funs[i](qdq) - pos_funs[j](qdq))

    min_dist_loc = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
    if dist_mat[min_dist_loc] <= 2*par.ball_radius:
        print("Collision between balls detected!")
        balls_collision = min_dist_loc
    else:
        balls_collision = None

    # -- Collisions with ground
    heights = np.abs(np.array((ball_pos(qdq)[1] for ball_pos in pos_funs)))
    min_height_loc = np.argmin(heights)

    if heights[min_height_loc] < par.ball_radius:
        print("Collision with ground detected")
        ground_collision = min_height_loc
    else:
        ground_collision = None

    return balls_collision, ground_collision


def emulate_sensor(y_in, threshold):

    # Expect a Boolean array
    finished = False
    p = 0.2
    i = 0
    y_out = np.abs(y_in) < threshold

    while not finished:
        i += geom.rvs(p) + 7
        if i < len(y_in):
            y_out[i] = ~y_out[i]
        else:
            finished = True

    return y_out


if __name__ == "__main__":
    test = np.random.random(size=20)*10
    test_faults = emulate_sensor(test, 7)
    for i in range(len(test)):
        print(f"{test[i]}  {test_faults[i]}")
