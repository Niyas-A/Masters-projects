import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Planner
import fcl
from scipy.spatial.transform import Rotation as R
import path_planner
from test import rrt_planner

def tic():
    return time.time()
def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))


def load_map(fname):
    '''
    Loads the bounady and blocks from map file fname.

    boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]

    blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
              ...,
              ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
    '''
    mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'), \
                                      'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
    blockIdx = mapdata['type'] == b'block'
    boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
    blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
    return boundary, blocks


def draw_map(boundary, blocks, start, goal):
    '''
    Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hb = draw_block_list(ax,blocks)
    hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
    hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(boundary[0,0],boundary[0,3])
    ax.set_ylim(boundary[0,1],boundary[0,4])
    ax.set_zlim(boundary[0,2],boundary[0,5])
    return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
    '''
    Subroutine used by draw_map() to display the environment blocks
    '''
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
    f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
    clr = blocks[:,6:]/255
    n = blocks.shape[0]
    d = blocks[:,3:6] - blocks[:,:3]
    vl = np.zeros((8*n,3))
    fl = np.zeros((6*n,4),dtype='int64')
    fcl = np.zeros((6*n,3))
    for k in range(n):
        vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
        fl[k*6:(k+1)*6,:] = f + k*8
        fcl[k*6:(k+1)*6,:] = clr[k,:]

    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
        pc.set_facecolor(fcl)
        h = ax.add_collection3d(pc)
        return h


def check_collision(path, blocks):
    collision = False
    for i in range(len(path) - 1):
        segment_start = np.array(path[i])
        segment_end = np.array(path[i + 1])
        direction = segment_end - segment_start
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        else:
            direction = np.zeros_like(direction)

        # Create a capsule for the path segment
        capsule = fcl.Capsule(radius=0.01, lz=length)
        T = (segment_start + segment_end) * 0.5
        z_axis = np.array([0, 0, 1])

        if np.allclose(direction, z_axis):
            q = R.from_quat([0, 0, 0, 1]).as_quat()  # Identity quaternion
        else:
            # Calculate the axis of rotation (cross product of z-axis and direction)
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            if rotation_axis_norm != 0:
                rotation_axis = rotation_axis / rotation_axis_norm

            # Calculate the angle of rotation (dot product of z-axis and direction)
            dot_product = np.dot(z_axis, direction)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            # Create the quaternion from the rotation axis and angle
            q = R.from_rotvec(rotation_axis * angle).as_quat()

        tf_capsule = fcl.Transform(q, T)
        O1 = fcl.CollisionObject(capsule, tf_capsule)

        for block in blocks:
            block_min = np.array(block[0:3])
            block_max = np.array(block[3:6])
            box_dims = block_max - block_min

            # Create a box for the AABB
            aabb = fcl.Box(*box_dims)
            T_aabb = block_min + box_dims * 0.5
            tf_aabb = fcl.Transform(T_aabb)

            O2 = fcl.CollisionObject(aabb, tf_aabb)

            # Check for collision
            request = fcl.CollisionRequest()
            result = fcl.CollisionResult()
            ret = fcl.collide(O1, O2, request, result)
            # print(ret)
            if ret > 0:
                collision = True
                break
        if collision:
            break
    return collision


def runtest(mapfile, start, goal, verbose = True):
    '''
    This function:
     * loads the provided mapfile
     * creates a motion planner
     * plans a path from start to goal
     * checks whether the path is collision free and reaches the goal
     * computes the path length as a sum of the Euclidean norm of the path segments
    '''
    # Load a map and instantiate a motion planner
    boundary, blocks = load_map(mapfile)
    MP = Planner.MyPlanner(boundary, blocks) # TODO: replace this with your own planner implementation

    # Display the environment
    if verbose:
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

    # Call the motion planner
    t0 = tic()
    path = MP.plan(start, goal)
    toc(t0,"Planning")

    # TODO: You should verify whether the path actually intersects any of the obstacles in continuous space
    # TODO: You can implement your own algorithm or use an existing library for segment and
    #       axis-aligned bounding box (AABB) intersection


    # print(path)
    print(start)
    print(goal)
    print(boundary[0])
    print(blocks)
    # print(type(path))
    path = path_planner.weighted_a_star(start, goal, blocks, boundary[0], epsilon=100)
    # path = rrt_planner(start, goal, boundary[0], blocks, max_samples = 100000)
    # print(type(path))
    collision = check_collision(path, blocks)
    print("collision :", collision)
    # collision = False
    goal_reached = sum((path[-1]-goal)**2) <= 0.1
    print("goal reached :", goal_reached)
    success = (not collision) and goal_reached
    pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))

    # Plot the path
    if verbose:
        # pass
        ax.plot(path[:,0],path[:,1],path[:,2],'r-')

    return success, pathlength


def test_single_cube(verbose = True):
    print('Running single cube test...\n')
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 5.5])
    success, pathlength = runtest('./maps/single_cube.txt', start, goal, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


def test_maze(verbose = True):
    print('Running maze test...\n')
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([12.0, 12.0, 5.0])
    success, pathlength = runtest('./maps/maze.txt', start, goal, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


def test_window(verbose = True):
    print('Running window test...\n')
    start = np.array([0.2, -4.9, 0.2])
    goal = np.array([6.0, 18.0, 3.0])
    success, pathlength = runtest('./maps/window.txt', start, goal, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


def test_tower(verbose = True):
    print('Running tower test...\n')
    start = np.array([2.5, 4.0, 0.5])
    goal = np.array([4.0, 2.5, 19.5])
    success, pathlength = runtest('./maps/tower.txt', start, goal, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


def test_flappy_bird(verbose = True):
    print('Running flappy bird test...\n')
    start = np.array([0.5, 2.5, 5.5])
    goal = np.array([19.0, 2.5, 5.5])
    success, pathlength = runtest('./maps/flappy_bird.txt', start, goal, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


def test_room(verbose = True):
    print('Running room test...\n')
    start = np.array([1.0, 5.0, 1.5])
    goal = np.array([9.0, 7.0, 1.5])
    success, pathlength = runtest('./maps/room.txt', start, goal, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


def test_monza(verbose = True):
    print('Running monza test...\n')
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])
    success, pathlength = runtest('./maps/monza.txt', start, goal, verbose)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')


if __name__=="__main__":
    test_single_cube() # working in 26, 67 with 0.2 and 100 path, 46 with 0.3 and 100 path 13
    test_maze() # working in 7786 with step=0.5 and 100, 146223 with 0.3 and 100 path 85 , long with 0.2 and 100 path
    test_flappy_bird() # working in 52206, 1436 with 0.3 and 100 path 33, 4335 with 0.2 and 100 path
    test_monza() # working in 52390 with 0.2 and 100 path , 22683 with step=0.3 and 100 path 81, 763513 with 1 and 580775 with 10 and 562026 with 100
    test_window() # working in 627, 146 with 0.3 and 100 path 32, 205 with 0.2 and 100 path
    test_tower() # working in 13931, 578 with 0.3 and 100 path 45, 1089 with 0.2 and 100 path
    test_room() # working in 12406 , 416 with 0.3 and 100 path 16, 3284 with 0.2 and 100 path 19
    plt.show(block=True)








