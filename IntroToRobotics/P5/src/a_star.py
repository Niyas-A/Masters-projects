import heapq
import matplotlib.pyplot as plt
import numpy as np
import random

grid_size = 11


def distance(start, goal):
    return np.sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2)


def valid_directions():
    return [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def inbounds(grid, point):
    x, y = point
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0


def is_new_node_lower_cost(new_cost, current_cost, neighbor):
    return neighbor not in current_cost or new_cost < current_cost[neighbor]


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def min_distance_to_obstacles(current, goal, obstacles):
    # Calculate the distance to the nearest obstacle
    # min_distance = min(np.linalg.norm(np.array(current) - np.array(obstacle)) for obstacle in obstacles)
    # return min_distance
    obstacle_proximity_weight = 10000000
    smoothness_weight = 0

    # Calculate obstacle proximity with quadratic cost increase
    min_distance_to_obstacle = min(
        np.linalg.norm(np.array(current) - np.array(obstacle)) for obstacle in obstacles
    )
    obstacle_cost = 1 / (min_distance_to_obstacle ** 2 + 1)  # Quadratic cost increase

    # Smoothness factor (encourage smooth paths)
    smoothness = 1 / (np.linalg.norm(np.array(current) - np.array(goal)) + 1)

    # Combine factors with weights
    safety_cost = (
        obstacle_proximity_weight * obstacle_cost
        + smoothness_weight * smoothness
        # Add more factors as needed
    )

    return safety_cost


def a_star(grid, start, goal, hueristic_name="distance"):
    directions = valid_directions()
    obstacles = [
        (x, y) for x in range(len(grid)) for y in range(len(grid[0])) if grid[x][y] == 1
    ]
    new_obstacles = []
    if hueristic_name == "safety":
        for (x, y) in obstacles:
            new_obstacles.append((x, y))
            for (dx, dy) in directions:
                point = (x + dx, y + dy)
                if point == start or point == goal:
                    continue
                if (x + dx, y + dy) in obstacles:
                    continue
                if 0 <= x + dx < grid_size and 0 <= y + dy < grid_size:
                    grid[x + dx][y + dy] = 3
                    new_obstacles.append((x + dx, y + dy))
    heap = [(0, start)]
    visited_path = {}
    current_cost = {start: 0}
    while heap:
        current_g_score, current = heapq.heappop(heap)
        if current == goal:
            return reconstruct_path(visited_path, current)

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if not inbounds(grid, neighbor):
                continue

            movement_cost = distance((dx, dy), (0, 0))
            new_cost = current_cost[current] + movement_cost

            if is_new_node_lower_cost(new_cost, current_cost, neighbor):
                current_cost[neighbor] = new_cost
                if hueristic_name == "distance":
                    priority = new_cost + distance(neighbor, goal)
                elif hueristic_name == "safety":
                    priority = (
                        new_cost
                        + min_distance_to_obstacles(neighbor, goal, obstacles)
                        + distance(neighbor, goal)
                    )

                else:
                    raise NotImplementedError("Not elemented hueristic")
                heapq.heappush(heap, (priority, neighbor))
                visited_path[neighbor] = current

    return []


def visualize_path(grid, path):
    # Create a colormap with better contrast
    pass
    # Create a legend for the colormap
    legend_labels = {0: "Open Position", 1: "Wall", 3: "Padding", 4: "Goal"}

    # Visualize the grid with a legend
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=4)
    fig, ax = plt.subplots()

    # Mark the path with a different value
    for point in path:
        grid[point[0]][point[1]] = 2

    # Visualize the grid with custom styling
    im = ax.imshow(grid, cmap=cmap, origin="lower", interpolation="none")

    # Create a legend and set its labels
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=legend_labels[label],
            markerfacecolor=cmap(norm(label)),
        )
        for label in legend_labels
    ]

    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    # Display the colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 3, 4], orientation="vertical")
    cbar.set_ticklabels([legend_labels[label] for label in [0, 1, 3, 4]])

    if path:
        # Highlight start and goal points with different colors
        ax.scatter(*path[0][::-1], color="blue", marker="o", s=100, label="Start")
        ax.scatter(*path[-1][::-1], color="orange", marker="o", s=100, label="Goal")

    # Add a legend
    plt.legend()

    # Draw a grid overlay
    plt.grid(color="white", linewidth=1)

    # Show the plot
    plt.title("A* Safety Hueristic")
    plt.show()


def create_obstacles_around_center(grid, center, num_obstacles):
    obstacles = set()

    while len(obstacles) < num_obstacles:
        x = random.randint(center[0] - 2, center[0] + 2)
        y = random.randint(center[1] - 2, center[1] + 2)

        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and (x, y) != center:
            obstacles.add((x, y))

    for obstacle in obstacles:
        grid[obstacle[0]][obstacle[1]] = 1  # Mark obstacles with a value of 1


def smart_atan(y, x):
    if x == 0 and y != 0:
        return np.pi / 2
    if y == 0:
        if x < 0:
            return np.pi
        return 0
    return np.arctan(y / x)


def reduce_wp(points):
    new_points = []
    new_points.append(points[0])
    prev_p = points[0]
    for p in points:
        if any([d1 == d2 for d1, d2 in zip(p, prev_p)]):
            prev_p = p
            continue
        if new_points[-1] != prev_p:
            new_points.append(prev_p)
        new_points.append(p)
        prev_p = p
    return new_points


# def wp():
def get_wp(scaling=4.0, heuristic=None):
    # waypoint = []
    # N = 11
    # for x in range(N):
    #     if x % 2 == 1:
    #         waypoint.append([x, N - 1])
    #         waypoint.append([x, 0])
    #     else:
    #         waypoint.append([x, 0])
    #         waypoint.append([x, N - 1])
    waypoints = []
    # grid 
    N = 11
    x, y = 0, 0
    max_x, max_y = N - 1, N - 1
    while True:
        waypoints.append([x, y])
        waypoints.append([x, max_y])
        waypoints.append([max_x, max_y])
        waypoints.append([max_x, y])
        waypoints.append([x, y])
        x += 1
        y += 1
        max_x -= 1
        max_y -= 1
        if x > max_x or y > max_y:
            break
    waypoint = [(y, x) for (x,y) in waypoints]
    new_waypoints = []
    prev_x, prev_y = waypoint[0][0], waypoint[0][1]
    for i in range(len(waypoints)):
        x, y = waypoint[i][0], waypoint[i][1]
        angle = smart_atan((y - prev_y), (x - prev_x))
        new_waypoints.append([x / scaling, y / scaling, angle])
        prev_x = x
        prev_y = y
    # waypoint = reduce_wp(new_waypoints)
    waypoint = np.array(new_waypoints)
    return waypoint


if __name__ == "__main__":
    get_wp(scaling=12.0, heuristic="distance")
