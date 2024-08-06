import numpy as np

from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.rrt.rrt_connect import RRTConnect
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot
from rrt_algorithms.rrt.rrt_star_bid import RRTStarBidirectional
from rrt_algorithms.rrt.rrt_star_bid_h import RRTStarBidirectionalHeuristic

# X_dimensions = np.array([(0, 100), (0, 100), (0, 100)])  # dimensions of Search Space
# # obstacles
# Obstacles = np.array(
#     [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
#      (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])
# x_init = (0, 0, 0)  # starting location
# x_goal = (100, 100, 100)  # goal location
def rrt_planner(start, goal, boundary, blocks, max_samples = 1024):
    # start = [2.3, 2.3, 1.3]
    # goal = [7.,  7.,  5.5]
    # boundary = [ -5.,  -5.,  -5.,  10.,  10.,  10., 120., 120., 120.]
    # blocks = [[  4.5,   4.5,   2.5,   5.5,   5.5,   3.5, 120.,  120.,  120. ]]
    X_dimensions = np.array([(boundary[0], boundary[3]), (boundary[1], boundary[4]), (boundary[2], boundary[5])])  # dimensions of Search Space
    print(X_dimensions)
    # obstacles
    Obstacles = np.array([tuple(block[0:6]) for block in blocks])
    print(Obstacles)
    x_init = tuple(start) #(0, 0, 0)  # starting location
    x_goal = tuple(goal) #(100, 100, 100)  # goal location
    print(x_init)
    print(x_goal)

    q = 1  # length of tree edges
    r = 0.05  # length of smallest edge to check for intersection with obstacles
    # max_samples = 1024  # max number of samples to take before timing out
    prc = 0.1  # probability of checking for a connection to goal
    rewire_count = 32  # optional, number of nearby branches to rewire

    # create Search Space
    X = SearchSpace(X_dimensions, Obstacles)

    # create rrt_search
    # rrt = RRTStarBidirectional(X, q, x_init, x_goal, max_samples, r, prc, rewire_count)
    # path = rrt.rrt_star_bidirectional()
    # rrt = RRTStarBidirectionalHeuristic(X, q, x_init, x_goal, max_samples, r, prc, rewire_count)
    # path = rrt.rrt_star_bid_h()
    # rrt_connect = RRTConnect(X, q, x_init, x_goal, max_samples, r, prc)
    # path = rrt_connect.rrt_connect()
    # rrt = RRTStar(X, q, x_init, x_goal, max_samples, r, prc, rewire_count)
    # path = rrt.rrt_star()
    rrt = RRT(X, q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()
    path_np = [list(point) for point in path]
    # print(path_np)
    return np.array(path_np)
    # plot
    # plot = Plot("rrt_3d")
    # plot.plot_tree(X, rrt.trees)
    # if path is not None:
    #     plot.plot_path(X, path)
    # plot.plot_obstacles(X, Obstacles)
    # plot.plot_start(X, x_init)
    # plot.plot_goal(X, x_goal)
    # plot.draw(auto_open=True)
