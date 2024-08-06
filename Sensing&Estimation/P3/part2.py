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

# (b) Landmark prior Mapping
print(features.shape)
f_set = set()
Z = np.zeros((3,features.shape[1]))
p = np.array([b, 0, 0])  # Translation vector
R = np.eye(3)
e3 = np.array([0, 0, 1])
for i in range(1,features.shape[2]):
# for i in range(1,2):
    f = features[:,:,i]
    missing_observation = np.array([-1, -1, -1, -1])
    missing_mask = np.all(f == missing_observation[:, np.newaxis], axis=0)
    non_missing_indices = np.where(~missing_mask)[0]
    for j in range(len(non_missing_indices)):
        if non_missing_indices[j] not in f_set:
            zt = f[:,non_missing_indices[j]]
            d = zt[0]-zt[2]
            if d>=1:
                z1 = zt[0:2]
                z2 = zt[2:]
                z1_h = np.linalg.inv(K)@np.append(z1, 1)
                z2_h = np.linalg.inv(K)@np.append(z2, 1)
                A = R.T @ p - (e3.T @ R.T @ p) * z2_h
                B = R.T @ z1_h - (e3.T @ R.T @ z1_h) * z2_h
                lambda1 = (A.T @ A) / (A.T @ B)
                m = lambda1 * z1_h
                m_new = Pose_imu[:,:,i]@imu_T_cam@np.append(m, 1)
                Z[:,non_missing_indices[j]] = m_new[:3]
                f_set.add(non_missing_indices[j])
print(len(f_set))

# (b) Landmark Mapping via EKF Update
print(features.shape)
f_set = set()
mu_m = np.zeros((3 * features.shape[1]))
cov_m = np.eye(3 * features.shape[1])
p = np.array([b, 0, 0])  # Translation vector
R = np.eye(3)
e3 = np.array([0, 0, 1])
V = 100000


def observation_model(m, Ks, imu_T_cam, rTw):
    m_new2 = np.linalg.inv(rTw @ imu_T_cam) @ np.append(m, 1)
    m_new3 = Ks @ m_new2[0:3] / m_new2[2]
    m_new4 = np.concatenate((m_new3[0:2], m_new3[0:2] - p[0:2]), axis=0)
    return m_new4


def compute_jacobian(m, Ks, imu_T_cam, rTw):
    P = np.hstack((np.eye(3), np.zeros((3, 1))))
    q = np.linalg.inv(rTw @ imu_T_cam) @ np.append(m, 1)
    # print(m_new)q
    q[0:3] = q[0:3] / q[2]
    dp_dq = np.eye(4)
    dp_dq[0, 2] = -q[0] / q[2]
    dp_dq[1, 2] = -q[1] / q[2]
    dp_dq[2, 2] = 0
    dp_dq[3, 2] = -q[3] / q[2]
    dp_dq = dp_dq / q[2]
    Ks_n = np.eye(4)
    Ks_n[:3, :3] = Ks
    h = Ks_n @ dp_dq @ np.linalg.inv(rTw @ imu_T_cam) @ P.T
    return h


# for i in range(1,500):
for i in range(1, features.shape[2]):
    z_t1 = []
    z_t1_bar = []
    obs_lst = []
    h_t1 = []
    f = features[:, :, i]
    missing_observation = np.array([-1, -1, -1, -1])
    missing_mask = np.all(f == missing_observation[:, np.newaxis], axis=0)
    non_missing_indices = np.where(~missing_mask)[0]
    for j in range(len(non_missing_indices)):
        if non_missing_indices[j] not in f_set:
            zt = f[:, non_missing_indices[j]]
            d = zt[0] - zt[2]
            if d >= 1:
                z1 = zt[0:2]
                z2 = zt[2:]
                z1_h = np.linalg.inv(K) @ np.append(z1, 1)
                z2_h = np.linalg.inv(K) @ np.append(z2, 1)
                A = R.T @ p - (e3.T @ R.T @ p) * z2_h
                B = R.T @ z1_h - (e3.T @ R.T @ z1_h) * z2_h
                lambda1 = (A.T @ A) / (A.T @ B)
                m = lambda1 * z1_h
                m_new = Pose_imu[:, :, i] @ imu_T_cam @ np.append(m, 1)
                # mu_m[:,non_missing_indices[j]] = m_new[:3]
                mu_m[3 * non_missing_indices[j]:3 * non_missing_indices[j] + 3] = m_new[:3]
                f_set.add(non_missing_indices[j])

                # print('obtained m:', m)
                # m_new2 = np.linalg.inv(Pose_imu[:,:,i]@imu_T_cam)@np.append(m_new[:3],1)
                # m_new3 = K@m_new2[0:3]/m_new2[2]
                # m_new4 = np.concatenate((m_new3[0:2], m_new3[0:2] - p[0:2]), axis=0)
                # print('regenerated z1: ',m_new4)
                # break
        else:
            # print(i)
            zt = f[:, non_missing_indices[j]]
            d = zt[0] - zt[2]
            if d >= 1:
                z1 = zt[0:2]
                z2 = zt[2:]
                obs_lst.append(non_missing_indices[j])
                z_t1.append(zt)
                z_t1_bar.append(
                    observation_model(mu_m[3 * non_missing_indices[j]:3 * non_missing_indices[j] + 3], K, imu_T_cam,
                                      Pose_imu[:, :, i]))
                h_t1.append(
                    compute_jacobian(mu_m[3 * non_missing_indices[j]:3 * non_missing_indices[j] + 3], K, imu_T_cam,
                                     Pose_imu[:, :, i]))
                # EKF Update:
    n = len(obs_lst)
    if n:
        # print(n)
        H_t1 = np.zeros((4 * n, 3 * n))
        for k, matrix in enumerate(h_t1):
            H_t1[4 * k:4 * (k + 1), 3 * k:3 * (k + 1)] = matrix
        z_t1_arr = np.hstack(z_t1)
        z_t1_bar_arr = np.hstack(z_t1_bar)
        sliced_list = [index for item in obs_lst for index in range(3 * item, 3 * item + 3)]
        cov_t = cov_m[np.ix_(sliced_list, sliced_list)]
        # mu_t_arr = np.hstack([mu_m[3*item:3*item+3] for item in obs_lst])
        mu_t_arr = mu_m[np.ix_(sliced_list)]
        I_V = V * np.eye(4 * n)
        K_t1 = cov_t @ H_t1.T @ np.linalg.inv(H_t1 @ cov_t @ H_t1.T + I_V)
        mu_t1 = mu_t_arr + K_t1 @ (z_t1_arr - z_t1_bar_arr)
        cov_t1 = (np.eye(3 * n) - K_t1 @ H_t1) @ cov_t
        mu_m[np.ix_(sliced_list)] = mu_t1
        cov_m[np.ix_(sliced_list, sliced_list)] = cov_t1

Z_ekf = mu_m.reshape(-1, 3)
import numpy as np
import matplotlib.pyplot as plt

plt.plot(Pose_imu[0, 3, :], Pose_imu[1, 3, :], color='red', linestyle='-', linewidth=2, label='EKF prediction Odometry')
plt.scatter(Z[0, :], Z[1, :], s=10, color='blue', alpha=0.6, label='Landmarks prior')
plt.scatter(Z_ekf[:, 0], Z_ekf[:, 1], s=10, color='black', alpha=0.6, label='Landmarks ekf')
plt.scatter(Pose_imu[0, 3, 0], Pose_imu[1, 3, 0], color='green', s=100, edgecolors='black', label='Start Position', zorder=5)
plt.scatter(Pose_imu[0, 3, -1], Pose_imu[1, 3, -1], color='orange', s=100, edgecolors='black', label='Stop Position', zorder=5)

plt.xlabel('X position (meters)', fontsize=12)
plt.ylabel('Y position (meters)', fontsize=12)
plt.title('Robot Position and Observed Landmarks', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best')
plt.gca().set_aspect('equal', adjustable='box')

plt.show()