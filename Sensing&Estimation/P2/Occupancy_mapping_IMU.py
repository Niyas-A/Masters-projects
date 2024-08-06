import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham2D

dataset = 21

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
    t = lidar_stamsp[i]
    index = np.argmin(np.abs(encoder_stamps - t))
    p_imu = pose_imu[0:3,index]

    # Filter valid ranges
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # Calculate positions in sensor frame
    xs0 = ranges * np.cos(angles)
    ys0 = ranges * np.sin(angles)

    # transform xs0 and ys0 using pose_imu
    x_imu, y_imu, theta_imu = p_imu
    xs_world = x_imu + xs0 * np.cos(theta_imu) - ys0 * np.sin(theta_imu)
    ys_world = y_imu + xs0 * np.sin(theta_imu) + ys0 * np.cos(theta_imu)

    # Convert positions to map frame
    xis = np.ceil((xs_world - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    yis = np.ceil((ys_world - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
    xis_imu = np.ceil((x_imu - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    yis_imu = np.ceil((y_imu - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    # Clamp indices within valid range
    xis = np.clip(xis, 0, MAP['sizex'] - 1)
    yis = np.clip(yis, 0, MAP['sizey'] - 1)

    # Build map
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)),
                                            (xis < MAP['sizex'])), (yis < MAP['sizey']))

    # Update log-odds for occupied cells
    log_odds_map[xis[indGood[0]], yis[indGood[0]]] += np.log(10)
    x_good = xis[indGood[0]]
    y_good = yis[indGood[0]]

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

plt.imshow(log_odds_map.T, cmap='binary', origin='lower')
plt.plot(np.ceil((pose_imu[0,:] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1, np.ceil((pose_imu[1,:] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1, color='red', linewidth=2)
plt.xlabel('X - grid')
plt.ylabel('Y - grid')
plt.title('Occupancy grid Map with IMU & Encoder odometry')
plt.show()