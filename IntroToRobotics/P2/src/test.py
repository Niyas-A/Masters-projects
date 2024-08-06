from pprint import pprint
import math
import csv

def smart_atan(x, y):
    if x == 0:
        if y > 0:
            result = math.pi / 2  # arctan(infinity) is +pi/2
        elif y < 0:
            result = -math.pi / 2  # arctan(-infinity) is -pi/2
        else:
            return 0
    else:
        result = math.atan(y / x)
        if x < 0 and y == 0:
            result += math.pi
        
    return result

with open("waypoints.txt") as f:
    rows = f.readlines()
    prev_x, prev_y, prev_theta = 0, 0, 0
    results = []
    for row in rows:
        x, y, theta = row.split(",")
        x, y, theta = float(x), float(y), float(theta)
        dist = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
        t2 = smart_atan(x - prev_x, y - prev_y)
        if x == 0 and y == 0 and theta == 0:
            results.append([0, 0, 0, 5])
            continue

        results.append([prev_x, prev_y, t2, abs(t2) * 4.5/(2*math.pi)])
        results.append([x, y, t2, 5.325/1.03 * dist])
        results.append([x, y, theta, abs(theta) * 4.5/(2*math.pi)])
        prev_x, prev_y, prev_theta = x, y, theta
    

    with open("new_waypoints.txt", "w") as f:
        wr = csv.writer(f)
        wr.writerows(results)

    #     if x == prev_x and y == prev_y and z == prev_z:
    #         results.append([x,y,z])
    #         prev_x, prev_y, prev_z = x, y, z

    #     if x - prev_x != 0:
    #         results.append([x, prev_y, prev_z])
    #         prev_x = x
        
    #     if z - prev_z != 0:
    #         results.append([prev_x, prev_y, z])
    #         prev_z = z
        
    #     if y - prev_y != 0:
    #         results.append([prev_x, y, prev_z])
    #         prev_y = y

    # pprint(results)
    # pprint(rows)