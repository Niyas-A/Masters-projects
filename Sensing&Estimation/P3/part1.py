import numpy as np
from pr3_utils import *

# Load the measurements
filename = "../data/03.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

# (a) IMU Localization via EKF Prediction
u = np.concatenate((linear_velocity,angular_velocity), axis=0)
u_hat = axangle2twist(u.T)
tau = np.diff(t)
tau = np.insert(tau, 0, 0)
from scipy.linalg import expm
T = np.eye(4)
Pose_imu = np.zeros((4,4,t.shape[1]))
S_imu = np.zeros((6,6,t.shape[1]))
u_chat = axangle2adtwist(u.T)
S = np.eye(6)*0.001 # covariance matrix
W = np.eye(6)*0.01
for i in range (1,t.shape[1]):
    T = T@expm(tau[i]*u_hat[i,:,:])
    S = expm(-tau[i]*u_chat[i,:,:])@S@expm(-tau[i]*u_chat[i,:,:]).T + W
    Pose_imu[:,:,i] = T
    S_imu[:,:,i] = S

import numpy as np
import matplotlib.pyplot as plt

plt.plot(Pose_imu[0,3,:], Pose_imu[1,3,:], color='red', linestyle='-', linewidth=2, label='EKF Prediction Odometry')
plt.scatter(Pose_imu[0, 3, 0], Pose_imu[1, 3, 0], color='green', s=100, edgecolors='black', label='Start Position', zorder=5)
plt.scatter(Pose_imu[0, 3, -1], Pose_imu[1, 3, -1], color='orange', s=100, edgecolors='black', label='Stop Position', zorder=5)

plt.xlabel('X position (meters)', fontsize=12)
plt.ylabel('Y position (meters)', fontsize=12)
plt.title('Robot Position', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()