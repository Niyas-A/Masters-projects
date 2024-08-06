from utils import *
import numpy as np

def stage_cost(x,u):
    if x[0] == x[5]:
        return 0
    else:
        return 1


def terminal_cost(x):
    if x[0] == x[5]:
        return 0
    else:
        return np.inf


def motion_model(current_state, action):
    position, orientation, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc = current_state

    if action == 0:  # Move Forward
        x_new = position[0] + orientation[0]
        y_new = position[1] + orientation[1]
        if x_new < 0 or y_new < 0 or x_new > grid_size-1 or y_new > grid_size-1:
            return current_state
        # door case
        elif (x_new,y_new) == door_loc[0]:
            if door1_open == 1:
                return ((x_new, y_new), orientation, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc)
            else:
                return current_state
        elif (x_new,y_new) == door_loc[1]:
            if door2_open == 1:
                return ((x_new, y_new), orientation, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc)
            else:
                return current_state
        # wall case
        if (x_new,y_new) in wall_loc:
            return current_state
        else:
            return ((x_new, y_new), orientation, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc)

    elif action == 1:  # Turn Left
        if orientation == (0,1):
            new_orientation = (1,0)
        elif orientation == (1,0):
            new_orientation = (0,-1)
        elif orientation == (0,-1):
            new_orientation = (-1,0)
        elif orientation == (-1,0):
            new_orientation = (0,1)
        return (position, new_orientation, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc)

    elif action == 2:  # Turn Right
        if orientation == (0, 1):
            new_orientation = (-1, 0)
        elif orientation == (1, 0):
            new_orientation = (0, 1)
        elif orientation == (0, -1):
            new_orientation = (1, 0)
        elif orientation == (-1, 0):
            new_orientation = (0, -1)
        return (position, new_orientation, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc)

    elif action == 3:  # Pickup Key
        x_new = position[0] + orientation[0]
        y_new = position[1] + orientation[1]
        if (x_new,y_new) == key_loc:
            has_key = 1
        return (position, orientation, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc)

    elif action == 4:  # Unlock Door
        # check if you are facing door
        x_new = position[0] + orientation[0]
        y_new = position[1] + orientation[1]
        if (x_new, y_new) == door_loc[0]:
            if has_key == 1:
                door1_open = (door1_open+1)%2
        elif (x_new, y_new) == door_loc[1]:
            if has_key == 1:
                door2_open = (door2_open+1)%2
        return (position, orientation, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc)

    else:
        # Unknown action, return current state
        return current_state

grid_size = 8
orientations = [(0, 1), (1, 0), (0, -1), (-1, 0)]
goals = [(5, 1), (6, 3), (5, 6)]
key_locs = [(1, 1), (2, 3), (1, 6)]
door_loc = ((4, 2), (4, 5))
wall_loc = ((4,0),(4,1),(4,3),(4,4),(4,6),(4,7))

states = [
    ((x, y), o, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc)
    for x in range(grid_size)
    for y in range(grid_size)
    for o in orientations
    for has_key in [0, 1]
    for door1_open in [0, 1]
    for door2_open in [0, 1]
    for goal in goals
    for key_loc in key_locs
]

T = 50
U = [0,1,2,3,4]

# Create dictionary
Vt = {state: terminal_cost(state) for state in states}
Vt_1 = {}

for t in range(T-1,-1,-1):
    # print('t = ',t)
    Qt = {}
    pt = {}

    for x in states:
        Qt[x] = {}
        for u in U:
            Qt[x][u] = stage_cost(x, u) + Vt[motion_model(x, u)]
        Vt_1[x] = min(Qt[x].values())

        pt[x] = min(Qt[x], key=Qt[x].get)

    if Vt_1 == Vt:
        pass
        # print('equal')
        # break

    Vt = Vt_1


for i in range(36):
    env_path = "./envs/random_envs/DoorKey-8x8-" + str(i+1) + ".env"
    env, info = load_env(env_path)
    print(info)
    #((x, y), o, has_key, door1_open, door2_open, goal, key_loc, door_loc, wall_loc)
    # door1_open = 1 if info["door_open"][0] else 0
    # door2_open = 1 if info["door_open"][1] else 0
    door1 = env.grid.get(4, 2)
    door2 = env.grid.get(4, 5)
    door1_open = door1.is_open
    door2_open = door2.is_open
    # print(door1_open, door2_open)
    has_key = 0
    initial_state = (tuple(info['init_agent_pos']),tuple(info['init_agent_dir']),has_key,door1_open,door2_open,tuple(info['goal_pos']),tuple(info['key_pos']),door_loc,wall_loc)
    # print(initial_state)
    # print(states[0])
    current_state = initial_state
    seq = []
    while current_state[0] != current_state[5]:
        action = pt[current_state]
        seq.append(action)
        print(action)
        current_state = motion_model(current_state,action)
    draw_gif_from_seq(seq, load_env(env_path)[0], path="./gif/random_maps/DoorKey-8x8-" + str(i+1) +".gif")
    create_grid_image_from_seq_with_gaps(seq, env, path="./grid_image/random_maps/DoorKey-8x8-" + str(i+1) +".png")

