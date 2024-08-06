import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from ICP import icp
import gtsam

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

n = encoder_counts.shape[1]
gtsam_pose = np.zeros((3,n), dtype=float)
graph = gtsam.NonlinearFactorGraph()
prior_model = gtsam.noiseModel.Diagonal.Sigmas((1e-6, 1e-6, 1e-4))
# prior_model = gtsam.noiseModel.Diagonal.Sigmas([0.0000001, 0.000001, 0.000001])
graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), prior_model))
odometry_model = gtsam.noiseModel.Diagonal.Sigmas((0.5, 0.5, 0.1))
loop_closure = gtsam.noiseModel.Diagonal.Sigmas([0.5, 0.5, 0.1])
loop_closure_end = gtsam.noiseModel.Diagonal.Sigmas([0.0001, 0.001, 0.0001])
# odometry_model = gtsam.noiseModel.Diagonal.Sigmas([0.001, 0.001, 0.001])
Between = gtsam.BetweenFactorPose2
initial_estimate = gtsam.Values()
initial_estimate.insert(0, gtsam.Pose2(0,0,0))

n = encoder_counts.shape[1]
for i in range(1,n):
    dx, dy, dtheta = pose_imu[0:3,i]-pose_imu[0:3,i-1]
    graph.add(gtsam.BetweenFactorPose2(i-1, i, gtsam.Pose2(dx, dy, dtheta), odometry_model))
    initial_estimate.insert(i, gtsam.Pose2(pose_imu[0,i], pose_imu[1,i], pose_imu[2,i]))
    interval = 5
    if i>interval-1:
        t1 = encoder_stamps[i]
        index = np.argmin(np.abs(lidar_stamsp - t1))
        lidar_range_t1 = lidar_ranges[:,index]
        X = lidar_range_t1 * np.cos(np.radians(angles))
        Y = lidar_range_t1 * np.sin(np.radians(angles))
        lidar_points_t1[0,:] = X.T
        lidar_points_t1[1,:] = Y.T
        pose_t1 = pose_imu[0:3,i]
        # t-1
        t = encoder_stamps[i-interval]
        index = np.argmin(np.abs(lidar_stamsp - t))
        lidar_range_t = lidar_ranges[:,index]
        X = lidar_range_t * np.cos(np.radians(angles))
        Y = lidar_range_t * np.sin(np.radians(angles))
        lidar_points_t[0,:] = X.T
        lidar_points_t[1,:] = Y.T
        pose_t = pose_imu[0:3,i-interval]
        theta = pose_t[2]
        wTt[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
        wTt[:3, 3] = np.array([pose_t[0], pose_t[1], 0])
        theta = pose_t1[2]
        wTt1[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
        wTt1[:3, 3] = np.array([pose_t1[0], pose_t1[1], 0])
        tTt1= np.linalg.inv(wTt)@wTt1
        T, distances, iter = icp(lidar_points_t1.T,lidar_points_t.T,tTt1)
        dx = T[0, 3]
        dy = T[1, 3]
        r = R.from_matrix(T[:3, :3])
        rot = r.as_euler('zyx')
        dtheta = rot[0]
        graph.add(gtsam.BetweenFactorPose2(i-interval, i, gtsam.Pose2(dx, dy, dtheta), loop_closure))
    # if i%1000 == 0:
    #     dx, dy, dtheta = pose[0:3,i]-pose[0:3,i-100]
    #     graph.add(gtsam.BetweenFactorPose2(i-10, i, gtsam.Pose2(dx, dy, dtheta), odometry_model))

# End closure factor
t1 = encoder_stamps[n-1]
index = np.argmin(np.abs(lidar_stamsp - t1))
lidar_range_t1 = lidar_ranges[:,index]
X = lidar_range_t1 * np.cos(np.radians(angles))
Y = lidar_range_t1 * np.sin(np.radians(angles))
lidar_points_t1[0,:] = X.T
lidar_points_t1[1,:] = Y.T
pose_t1 = pose_imu[0:3,n-1]
# t-1
t = encoder_stamps[0]
index = np.argmin(np.abs(lidar_stamsp - t))
lidar_range_t = lidar_ranges[:,index]
X = lidar_range_t * np.cos(np.radians(angles))
Y = lidar_range_t * np.sin(np.radians(angles))
lidar_points_t[0,:] = X.T
lidar_points_t[1,:] = Y.T
pose_t = pose_imu[0:3,0]
theta = pose_t[2]
wTt[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
wTt[:3, 3] = np.array([pose_t[0], pose_t[1], 0])
theta = pose_t1[2]
wTt1[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
wTt1[:3, 3] = np.array([pose_t1[0], pose_t1[1], 0])
tTt1= np.linalg.inv(wTt)@wTt1
T, distances, iter = icp(lidar_points_t1.T,lidar_points_t.T,tTt1)
dx = T[0, 3]
dy = T[1, 3]
r = R.from_matrix(T[:3, :3])
rot = r.as_euler('zyx')
dtheta = rot[0]
graph.add(gtsam.BetweenFactorPose2(0, n-1, gtsam.Pose2(dx, dy, dtheta), loop_closure_end))

# Optimize the initial values using a Gauss-Newton nonlinear optimizer
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
result = optimizer.optimize()

for i in range(0,n):
    pose_2d = result.atPose2(i)
    gtsam_pose[:,i] = [pose_2d.x(),pose_2d.y(),pose_2d.theta()]

plt.plot(factor_pose[0,:], factor_pose[1,:], label='GTSAM Interval 5 and end closure')
plt.plot(pose_icp[0,:], pose_icp[1,:], label='Lidar ICP Odometry')
plt.plot(pose_imu[0,:], pose_imu[1,:], label='IMU & Encoder Odometry')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Robot position')
plt.grid(True)
plt.legend()  # Add legends
plt.show()