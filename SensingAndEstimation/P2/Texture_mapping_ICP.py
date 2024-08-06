import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
from ICP import icp

dataset = 20

with np.load("../data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load("../data/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans

with np.load("../data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

with np.load("../data/Kinect%d.npz"%dataset) as data:
  disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
  rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

n = encoder_counts.shape[1]
pose_imu = np.zeros((4,n), dtype=float)
m_per_tick = np.pi*0.254/360
for i in range(1,n):
    tau = encoder_stamps[i] - encoder_stamps[i-1]
    # encoder_count = encoder_counts[:,i] - encoder_counts[:,i-1]
    encoder_count = encoder_counts[:,i]
    # encoder_count = [FR, FL, RR, RL]
    vl = m_per_tick*(encoder_count[1]+encoder_count[3])/2
    vr = m_per_tick*(encoder_count[0]+encoder_count[2])/2
    vt = (vr + vl)/2
    theta = pose_imu[2,i-1]
    t = encoder_stamps[i-1]
    index = np.argmin(np.abs(imu_stamps - t))
    omega = imu_angular_velocity[2,index]
    pose_imu[0:3,i] = pose_imu[0:3,i-1] + np.array([vt*np.cos(theta), vt*np.sin(theta), tau*omega])
    pose_imu[3,i] = encoder_stamps[i]

lidar_range = lidar_ranges[:,1]
angles = np.linspace(-135, 135, len(lidar_range))
lidar_points_t = np.zeros((3,len(lidar_range)), dtype=float)
lidar_points_t1 = np.zeros((3,len(lidar_range)), dtype=float)
n = len(lidar_stamsp)
pose_icp = np.zeros((3,n), dtype=float)
# pose_icp[3, :] = 1
wTt = np.eye(4, dtype=float)
wTt1 = np.eye(4, dtype=float)
tTt1 = np.eye(4, dtype=float)
prevT = np.eye(4, dtype=float)
Pose_icp = np.zeros((4,4,n), dtype=float)
# i = 1000
for i in range(1,n):
    # t
    lidar_range_t1 = lidar_ranges[:,i]
    X = lidar_range_t1 * np.cos(np.radians(angles))
    Y = lidar_range_t1 * np.sin(np.radians(angles))
    lidar_points_t1[0,:] = X.T
    lidar_points_t1[1,:] = Y.T
    # lidar_points_t1[3, :] = 1
    t1 = lidar_stamsp[i]
    index = np.argmin(np.abs(encoder_stamps - t1))
    pose_t1 = pose_imu[0:3,index]
    # t-1
    lidar_range_t = lidar_ranges[:,i-1]
    X = lidar_range_t * np.cos(np.radians(angles))
    Y = lidar_range_t * np.sin(np.radians(angles))
    lidar_points_t[0,:] = X.T
    lidar_points_t[1,:] = Y.T
    # lidar_points_t[3, :] = 1
    t = lidar_stamsp[i-1]
    index = np.argmin(np.abs(encoder_stamps - t))
    pose_t = pose_imu[0:3,index]
    theta = pose_t[2]
    wTt[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
    wTt[:3, 3] = np.array([pose_t[0], pose_t[1], 0])
    theta = pose_t1[2]
    wTt1[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
    wTt1[:3, 3] = np.array([pose_t1[0], pose_t1[1], 0])
    tTt1= np.linalg.inv(wTt)@wTt1
    # ICP_new3()
    T, distances, iter = icp(lidar_points_t1.T,lidar_points_t.T,tTt1)
    prevT = prevT@T
    pose_icp[:,i] = prevT[:3, 3]
    Pose_icp[:,:,i] = prevT

disp_path = "../data/Disparity"+str(dataset)+"/"
rgb_path = "../data/RGB"+str(dataset)+"/"
# print(rgb_path)

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

# init MAP
MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -30  #meters
MAP['ymin']  = -30
MAP['xmax']  =  30
MAP['ymax']  =  30
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
map_rgb = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)
oRr = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
camera_position = np.array([0.18, 0.005, 0.36])
roll = 0.0  # in radians
pitch = 0.36  # in radians
yaw = 0.021  # in radians
rotation_matrix = np.array([
    [np.cos(yaw) * np.cos(pitch), -np.sin(yaw) * np.cos(roll) + np.cos(yaw) * np.sin(pitch) * np.sin(roll), np.sin(yaw) * np.sin(roll) + np.cos(yaw) * np.sin(pitch) * np.cos(roll)],
    [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(roll) + np.sin(yaw) * np.sin(pitch) * np.sin(roll), -np.cos(yaw) * np.sin(roll) + np.sin(yaw) * np.sin(pitch) * np.cos(roll)],
    [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
])
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rotation_matrix
transformation_matrix[:3, 3] = camera_position

n = len(rgb_stamps)
print(n)
for im in range(1,n):
    if im%100 == 0:
        print(im)
    # load RGBD image
    imd = cv2.imread(disp_path+'disparity'+str(dataset)+'_'+str(im)+'.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    imc = cv2.imread(rgb_path+'rgb'+str(dataset)+'_'+str(im)+'.png')[...,::-1] # (480 x 640 x 3)

    # print(imc.shape)

    # convert from disparity from uint16 to double
    disparity = imd.astype(np.float32)

    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd

    # calculate u and v coordinates
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

    # get 3D coordinates
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])

    x_column = x.reshape(-1, 1)
    y_column = y.reshape(-1, 1)
    z_column = z.reshape(-1, 1)

    # Stack x, y, and z column vectors into a single matrix
    coordinates = np.hstack((x_column, y_column, z_column))
    r_coordinates = np.dot(coordinates, oRr)
    # b_coordinates = np.dot(transformation_matrix, np.hstack((r_coordinates, np.ones((r_coordinates.shape[0], 1)))).T)[:3, :].T
    b_coordinates = np.dot(np.linalg.inv(transformation_matrix), np.hstack((r_coordinates, np.ones((r_coordinates.shape[0], 1)))).T)[:3, :].T
    # b_coordinates = np.dot(np.hstack((r_coordinates, np.ones((r_coordinates.shape[0], 1)))), transformation_matrix)[:3, :].T

    # body to world
    xs0, ys0, zs0  = b_coordinates.T
    t = rgb_stamps[im]
    index = np.argmin(np.abs(encoder_stamps - t))
    # print(index)
    # p_imu = pose_imu[0:3,index]
    # x_imu, y_imu, theta_imu = p_imu
    # print(theta_imu)
    t = rgb_stamps[im]
    index = np.argmin(np.abs(lidar_stamsp - t))
    x_icp = Pose_icp[0,3,index]
    y_icp = Pose_icp[1,3,index]
    r = R.from_matrix(Pose_icp[:3, :3, index])
    rot = r.as_euler('zyx')
    theta_icp = rot[0]
    # print(theta_icp)
    xs_world = x_icp + xs0 * np.cos(theta_icp) - ys0 * np.sin(theta_icp)
    ys_world = y_icp + xs0 * np.sin(theta_icp) + ys0 * np.cos(theta_icp)

    # Convert positions to map frame
    xis = np.ceil((xs_world - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    yis = np.ceil((ys_world - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
    rgb_reshaped = imc.reshape(-1, 3)

    # for i in range(len(xis)):
    indGood = np.logical_and(np.logical_and(np.logical_and(np.logical_and(
        np.logical_and((zs0 > -0.5),(zs0 < 0.5)),(xis > 1)), (yis > 1)),(xis < MAP['sizex']))
        , (yis < MAP['sizey']))
    # map_rgb[xis[indGood[0]], yis[indGood[0]]] = rgb_reshaped[indGood[0]]

    # Extract indices where indGood is True
    valid_indices = np.where(indGood)[0]

    # Use these valid indices to access elements in xis, yis, and rgb_reshaped
    valid_xis = xis[valid_indices]
    valid_yis = yis[valid_indices]
    valid_rgb = rgb_reshaped[valid_indices]

    # Ensure the valid indices are within bounds of the map
    valid_xis = np.clip(valid_xis, 0, MAP['sizex'] - 1)
    valid_yis = np.clip(valid_yis, 0, MAP['sizey'] - 1)
    # valid_yis = MAP['sizey'] - 1 - valid_yis


    # Update map_rgb with valid indices
    map_rgb[valid_xis, valid_yis] = valid_rgb
    # map_rgb[valid_yis, valid_xis] = valid_rgb

plt.figure()
plt.imshow(map_rgb)
icp_pose_x = np.ceil((pose_icp[1, :] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
icp_pose_y = np.ceil((pose_icp[0, :] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
plt.plot(icp_pose_x, icp_pose_y, color='red', linewidth=2)
plt.xlabel('Y - grid')
plt.ylabel('X - grid')
plt.title('Texture Map with Lidar ICP odometry')
plt.show()