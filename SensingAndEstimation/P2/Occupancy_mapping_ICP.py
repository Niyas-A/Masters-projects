import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham2D
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

# Initialize MAP
MAP = {
    'res': 0.05,  # meters
    'xmin': -40,  # meters
    'ymin': -40,
    'xmax': 40,
    'ymax': 40,
}

MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

# log odds map
log_odds_map = np.zeros((MAP['sizex'], MAP['sizey']))

n = len(lidar_stamsp)
print(n)
for i in range(0,n):
    if i%100 == 0:
        print(i)
    angles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0
    ranges = lidar_ranges[:, i]
    # Filter valid ranges
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # Calculate positions in sensor frame
    xs0 = ranges * np.cos(angles)
    ys0 = ranges * np.sin(angles)

    # transform xs0 and ys0 using pose_imu
    x_icp = Pose_icp[0,3,i]
    y_icp = Pose_icp[1,3,i]
    r = R.from_matrix(Pose_icp[:3, :3, i])
    rot = r.as_euler('zyx')
    theta_icp = rot[0]
    xs_world = x_icp + xs0 * np.cos(theta_icp) - ys0 * np.sin(theta_icp)
    ys_world = y_icp + xs0 * np.sin(theta_icp) + ys0 * np.cos(theta_icp)

    # Convert positions to map frame
    xis = np.ceil((xs_world - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    yis = np.ceil((ys_world - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    xis_imu = np.ceil((x_icp - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    yis_imu = np.ceil((y_icp - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    # Clamp indices within valid range
    xis = np.clip(xis, 0, MAP['sizex'] - 1)
    yis = np.clip(yis, 0, MAP['sizey'] - 1)

    # Build map
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)),
                                            (xis < MAP['sizex'])), (yis < MAP['sizey']))

    # Update log-odds for occupied cells
    log_odds_map[xis[indGood[0]], yis[indGood[0]]] += np.log(8)

    x_good = xis[indGood[0]]
    y_good = yis[indGood[0]]
    # print(x_good.shape[1])
    for j in range(x_good.shape[1]):
        ray_cells = bresenham2D(xis_imu, yis_imu, x_good[0,j], y_good[0,j])
        x_cell, y_cell = ray_cells
        x_cell = np.clip(x_cell, 0, MAP['sizex'] - 1)
        y_cell = np.clip(y_cell, 0, MAP['sizey'] - 1)
        x_cell = x_cell.astype(int)
        y_cell = y_cell.astype(int)
        log_odds_map[x_cell, y_cell] -= np.log(4)

# log_odds_map = 1/(1 + np.exp(-log_odds_map))
clipped_log_odds_map = np.clip(log_odds_map, -50, 50)
exp_log_odds = np.exp(-clipped_log_odds_map) + 1
log_odds_map = 1/exp_log_odds

# Plot log-odds map
plt.figure()
# plt.imshow(log_odds_map.T, 'Greys')
plt.imshow(log_odds_map.T, cmap='binary', origin='lower')
plt.plot(np.ceil((pose_icp[0,:] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1, np.ceil((pose_icp[1,:] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1, color='red', linewidth=2)
plt.xlabel('X - grid')
plt.ylabel('Y - grid')
plt.title('Occupancy grid Map with Lidar ICP odometry')
plt.show()