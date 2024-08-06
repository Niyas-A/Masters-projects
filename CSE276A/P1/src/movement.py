import numpy as np
waypoints = []
with open("waypoints.txt", "r") as f:
    waypoints = f.readlines()
    waypoints = [waypoint.replace("\n", "").split(",") for waypoint in waypoints]
    waypoints = [list(map(float, item)) for item in waypoints]

r = 1 * 2.54  / 100
ly = 4.5 * 2.54  / 100

lx = 6 * 2.54 / 100

t = 10

for x, y, theta in waypoints:
    vx = x / t
    vy = y / t
    wz = theta / t

    result = 1 / r * np.array([
        [ 1, -1, -(lx + ly)],
        [ 1, 1, (lx + ly)],
        [ 1, 1, -(lx + ly)],
        [ 1, -1, (lx + ly)],
        ]).dot(np.array([vx, vy, wz]))
    speeds = list(result)
    print(speeds)