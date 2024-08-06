from collision_checker import check_collision
import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def boundary_check(point, boundary):
    x, y, z = point
    x_min, y_min, z_min = boundary[0:3]
    x_max, y_max, z_max = boundary[3:6]

    if x < x_min or x > x_max:
        return True
    if y < y_min or y > y_max:
        return True
    if z < z_min or z > z_max:
        return True

    return False

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def weighted_a_star(start, goal, obstacles, boundary, epsilon=1.2):
    # Heuristic function h
    def h(point):
        return euclidean_distance(point, goal)

    # Nodes - (f, g, position), f = g + epsilon*h
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)
    open_set = []
    heapq.heappush(open_set, (epsilon * h(start), 0, start_tuple))
    came_from = {}
    cost_so_far = {start_tuple: 0}
    closed_set = set()
    count = 0
    while open_set:
        current_f, current_g, current_tuple = heapq.heappop(open_set)
        current = np.array(current_tuple)

        if current_tuple in closed_set:
            continue

        if np.allclose(current, goal, atol=0.1):
            # reached goal
            print(count)
            return reconstruct_path(came_from, current_tuple)

        closed_set.add(tuple(current))
        next = None
        for next in generate_successors(current):
            # print(boundary_check(next, boundary))
            if not boundary_check(next, boundary):
                next_tuple = tuple(next)
                if tuple(next) in closed_set or check_collision(np.array([current, next]), obstacles):
                    continue
                new_cost = current_g + euclidean_distance(current, next)

                if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
                    cost_so_far[next_tuple] = new_cost
                    priority = new_cost + epsilon * h(next)
                    heapq.heappush(open_set, (priority, new_cost, next_tuple))
                    came_from[next_tuple] = tuple(current)
        count += 1
        if count % 1000 == 0:
            pass
            # print(next)
            # print(count)
    return None  # No path found


def generate_successors(node):
    # steps = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]), np.array([0, 0, 1]),
    #          np.array([0, 0, -1])]
    step = 0.3
    steps = [np.array([step, 0, 0]), np.array([-step, 0, 0]), np.array([0, step, 0]), np.array([0, -step, 0]), np.array([0, 0, step]),
             np.array([0, 0, -step])]
    return [node + step for step in steps]


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return np.array(path)


# def visualize(start, goal, path, obstacles):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     ax.scatter(*start, color='green', s=100, label='Start')
#     ax.scatter(*goal, color='red', s=100, label='Goal')
#
#     # Plot path
#     if path is not None:
#         ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', label='Path')
#
#     # Plot obstacles
#     for obstacle in obstacles:
#         min_corner = obstacle[:3]
#         max_corner = obstacle[3:]
#         x = [min_corner[0], max_corner[0]]
#         y = [min_corner[1], max_corner[1]]
#         z = [min_corner[2], max_corner[2]]
#         for xi, yi, zi in zip(x, y, z):
#             ax.bar3d(xi, yi, z[0], max_corner[0] - min_corner[0], max_corner[1] - min_corner[1],
#                      max_corner[2] - min_corner[2], color='black', alpha=0.5)
#
#     ax.legend()
#     plt.show()


# Example usage
# start = np.array([0, 0, 0])
# goal = np.array([10, 10, 10])
# obstacles = [np.array([4, 4, 4, 7, 7, 7]),
#              np.array([5, 8, 5, 6, 9, 6])]  # Define obstacles as [min_x, min_y, min_z, max_x, max_y, max_z]
# boundary = [np.array([-15, -15, 0, 15, 15, 15])]  # Define obstacles as [min_x, min_y, min_z, max_x, max_y, max_z]
# path = weighted_a_star(start, goal, obstacles, boundary[0], epsilon=10)
# print(path)
# visualize(start, goal, path, obstacles)