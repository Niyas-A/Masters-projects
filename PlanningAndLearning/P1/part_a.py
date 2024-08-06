from utils import *
import numpy as np


def stage_cost(x, u):
    if x[0] == x[4]:
        return 0
    else:
        return 1


def terminal_cost(x):
    if x[0] == x[4]:
        return 0
    else:
        return np.inf


def motion_model(current_state, action):
    position, orientation, has_key, door_open, goal, key_loc, door_loc = current_state

    if action == 0:  # Move Forward
        x_new = position[0] + orientation[0]
        y_new = position[1] + orientation[1]
        if x_new < 0 or y_new < 0 or x_new > grid_size - 1 or y_new > grid_size - 1:
            return current_state
        # door case
        elif (x_new, y_new) == door_loc:
            if door_open == 1:
                return ((x_new, y_new), orientation, has_key, door_open, goal, key_loc, door_loc)
            else:
                return current_state
        # wall case
        cell = env.grid.get(x_new, y_new)  # NoneType, Wall, Key, Goal
        if cell != None:
            if cell.type == "wall":
                return current_state
            elif cell.type == "door":  # remove this
                if door_open == 1:
                    return ((x_new, y_new), orientation, has_key, door_open, goal, key_loc, door_loc)
                else:
                    return current_state
            else:
                return ((x_new, y_new), orientation, has_key, door_open, goal, key_loc, door_loc)
        else:
            return ((x_new, y_new), orientation, has_key, door_open, goal, key_loc, door_loc)

    elif action == 1:  # Turn Left
        if orientation == (0, 1):
            new_orientation = (1, 0)
        elif orientation == (1, 0):
            new_orientation = (0, -1)
        elif orientation == (0, -1):
            new_orientation = (-1, 0)
        elif orientation == (-1, 0):
            new_orientation = (0, 1)
        return (position, new_orientation, has_key, door_open, goal, key_loc, door_loc)

    elif action == 2:  # Turn Right
        if orientation == (0, 1):
            new_orientation = (-1, 0)
        elif orientation == (1, 0):
            new_orientation = (0, 1)
        elif orientation == (0, -1):
            new_orientation = (1, 0)
        elif orientation == (-1, 0):
            new_orientation = (0, -1)
        return (position, new_orientation, has_key, door_open, goal, key_loc, door_loc)

    elif action == 3:  # Pickup Key
        x_new = position[0] + orientation[0]
        y_new = position[1] + orientation[1]
        if (x_new, y_new) == key_loc:
            has_key = 1
        return (position, orientation, has_key, door_open, goal, key_loc, door_loc)

    elif action == 4:  # Unlock Door
        # check if you are facing door
        x_new = position[0] + orientation[0]
        y_new = position[1] + orientation[1]
        if (x_new, y_new) == door_loc:
            if has_key == 1:
                door_open = (door_open + 1) % 2
        return (position, orientation, has_key, door_open, goal, key_loc, door_loc)

    else:
        # Unknown action, return current state
        return current_state


env_paths=[
        "doorkey-5x5-normal.env",
        "doorkey-6x6-normal.env",
        "doorkey-8x8-normal.env",
        "doorkey-6x6-direct.env",
        "doorkey-8x8-direct.env",
        "doorkey-6x6-shortcut.env",
        "doorkey-8x8-shortcut.env"
]
# i = 6
# for i in range(1):
for i in range(len(env_paths)):
    env_path = "./envs/known_envs/"+env_paths[i]
    env, info = load_env(env_path)  # load an environment
    print(info)

    # Define the size of the grid and the possible orientations
    grid_size = info['height'] #5 # This means a 2x2 grid
    orientations = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # List of orientation vectors
    goal = tuple(info['goal_pos']) #(3,3)
    key_loc = tuple(info['key_pos']) #(1,1)
    door_loc = tuple(info['door_pos']) #(2,2)

    # Generate all combinations of positions, orientations, isKey, and doorOpen
    states = [
        ((x, y), o, has_key, door_open, goal, key_loc, door_loc)
        for x in range(grid_size)
        for y in range(grid_size)
        for o in orientations
        for has_key in [0, 1]
        for door_open in [0, 1]
    ]

    print(len(states))

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

    #((x, y), o, has_key, door_open, goal, key_loc, door_loc)
    door = env.grid.get(info["door_pos"][0], info["door_pos"][1])
    is_open = door.is_open
    is_carrying = env.carrying is not None
    has_key = 1 if is_carrying else 0
    door_open = 1 if is_open else 0
    initial_state = (tuple(info['init_agent_pos']),tuple(info['init_agent_dir']),has_key,door_open,tuple(info['goal_pos']),tuple(info['key_pos']),tuple(info['door_pos']))
    # print(initial_state)
    current_state = initial_state
    seq = []
    while current_state[0]!=current_state[4]:
        action = pt[current_state]
        seq.append(action)
        # print(action)
        current_state = motion_model(current_state,action)
    print(env_paths[i].rsplit('.', 1)[0])
    print(seq)
    seq_string = convert_sequence(seq)
    print(seq_string)
    draw_gif_from_seq(seq, load_env(env_path)[0],path="./gif/known_maps/"+env_paths[i].rsplit('.', 1)[0]+".gif")
    create_grid_image_from_seq_with_gaps(seq, env,path="./grid_image/known_maps/"+env_paths[i].rsplit('.', 1)[0]+".png")
