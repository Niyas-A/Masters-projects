from utils import *
import numpy as np

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def stage_cost(x,u):
    if x[0] == x[4]:
        return 0
    else:
        return 1


def terminal_cost(x):
    if x[0] == x[4]:
        return 0
    else:
        return np.inf


def motion_model(x,u):

    return 0



def doorkey_problem(env):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """

    # optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    optim_act_seq = [TR, TR, PK, TR, UD, MF, MF, TR, MF]
    return optim_act_seq


def partA():
    env_path = "./envs/known_envs/doorkey-5x5-normal.env"
    # print("hi")
    env, info = load_env(env_path)  # load an environment
    print(info)
    # plot_env(env)
    # Get the agent position
    agent_pos = env.agent_pos
    print("agent_pos",agent_pos)

    # Get the agent direction
    agent_dir = env.dir_vec  # or env.agent_dir
    print("agent_dir",agent_dir)

    # Get the cell in front of the agent
    front_cell = env.front_pos  # == agent_pos + agent_dir
    print("front_cell",front_cell)

    # Access the cell at coord: (2,3)
    cell = env.grid.get(0, 0)  # NoneType, Wall, Key, Goal
    print('cell: ',cell)
    print(dir(cell))
    print(cell.type)
    cell = env.grid.get(1, 1)  # NoneType, Wall, Key, Goal
    print(cell.type)
    print(cell.type == "key")
    cell = env.grid.get(2, 2)  # NoneType, Wall, Key, Goal
    print(cell.type)
    cell = env.grid.get(3, 3)  # NoneType, Wall, Key, Goal
    if cell != None:
        print(cell.type)
    cell = env.grid.get(1, 2)  # NoneType, Wall, Key, Goal
    if cell != None:
        print(cell.type)

    # Get the door status
    door = env.grid.get(info["door_pos"][0], info["door_pos"][1])
    is_open = door.is_open
    is_locked = door.is_locked
    print('is_open: ',is_open)
    print('is_locked: ',is_locked)

    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None
    print('is_carrying: ',is_carrying)

    agent_dir = env.dir_vec  # or env.agent_dir
    print("agent_dir",agent_dir)
    cost, done = step(env, TR)
    agent_dir = env.dir_vec  # or env.agent_dir
    print("agent_dir",agent_dir)
    cost, done = step(env, TR)
    agent_dir = env.dir_vec  # or env.agent_dir
    print("agent_dir",agent_dir)
    cost, done = step(env, TR)
    agent_dir = env.dir_vec  # or env.agent_dir
    print("agent_dir",agent_dir)
    cost, done = step(env, TR)
    agent_dir = env.dir_vec  # or env.agent_dir
    print("agent_dir",agent_dir)


    # Take actions
    # cost, done = step(env, MF)  # MF=0, TL=1, TR=2, PK=3, UD=4
    # print("Moving Forward Costs: {}".format(cost))
    # cost, done = step(env, TL)  # MF=0, TL=1, TR=2, PK=3, UD=4
    # print("Turning Left Costs: {}".format(cost))
    # cost, done = step(env, TR)  # MF=0, TL=1, TR=2, PK=3, UD=4
    # print("Turning Right Costs: {}".format(cost))
    # cost, done = step(env, PK)  # MF=0, TL=1, TR=2, PK=3, UD=4
    # print("Picking Up Key Costs: {}".format(cost))
    # cost, done = step(env, UD)  # MF=0, TL=1, TR=2, PK=3, UD=4
    # print("Unlocking Door Costs: {}".format(cost))
    # print(info)
    # print(type(info))
    # print(info['init_agent_pos'])
    # print(info['init_agent_dir'])
    # print(info['door_pos'])
    # print(info['key_pos'])
    # print(info['goal_pos'])

    seq = doorkey_problem(env)  # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save


def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)
    plot_env(env)


if __name__ == "__main__":
    # example_use_of_gym_env()
    # partA()
    partB()

