import pickle
import sys
import time
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import math
import matplotlib.pyplot as plt

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

dataset="10"
ifile = "../testset/imu/imuRaw" + dataset + ".p"
ts = tic()
imud = read_data(ifile)
toc(ts,"Data import")
B = imud['vals']
Bt = imud['ts']

# IMU calibration
# Ax Ay Az Wz Wx Wy
# Ay and Ay are opposite
# Ax, Ay and Az: Sensitivity at XOUT, YOUT, ZOUT 270 300 330 mV/g Min Typ Max
imu_vals = np.empty_like(B, dtype=float)
bias = np.mean(B[:,0:50], axis=1)
# print(bias)
# Vref = 3000
Vref = 3300
sensitivity = 300
scale_factor = Vref / (1023 * sensitivity)
bias[2] = bias[2] - (1/scale_factor) # adjusting bias for 1g for acc_Z
# value = (raw - bias)*scale_factor
imu_vals[0,:] = (B[0,:]-bias[0])*scale_factor # acc_X
imu_vals[1,:] = (B[1,:]-bias[1])*scale_factor # acc_Y
imu_vals[2,:] = (B[2,:]-bias[2])*scale_factor # acc_Z
# Vref = 1230
Vref = 3300
sensitivity = 3.3 * 180 / math.pi
scale_factor = Vref / (1023 * sensitivity)
imu_vals[3,:] = -(B[4,:]-bias[4])*scale_factor # gyr_X
imu_vals[4,:] = -(B[5,:]-bias[5])*scale_factor # gyr_Y
imu_vals[5,:] = -(B[3,:]-bias[3])*scale_factor # gyr_Z
# print(imu_vals)
ts = Bt - Bt[0,0]
# print(ts)

# remove nan by adding e-6 in norm calculation
@jax.jit
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.T  
    w2, x2, y2, z2 = q2.T  
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z]).T  

@jax.jit
def quaternion_inverse(q):
    w, x, y, z = q.T  
    norm2 = w*w + x*x + y*y + z*z + 0.000001 
    return jnp.array([w/norm2, -x/norm2, -y/norm2, -z/norm2]).T  

@jax.jit
def quaternion_exp(q):
    w, x, y, z = q
    norm_xyz = jnp.linalg.norm(q[1:]) + 0.000001
    W = jnp.exp(w) * jnp.cos(norm_xyz)
    XYZ = jnp.exp(w) * jnp.sin(norm_xyz) * q[1:] / norm_xyz #if norm_xyz != 0 else jnp.zeros(3)
    return jnp.array([W, XYZ[0], XYZ[1], XYZ[2]])

@jax.jit
def quaternion_log(q):
    w, x, y, z = q.T  
    norm = jnp.sqrt(w*w + x*x + y*y + z*z) + 0.000001
    normXYZ = jnp.sqrt(x*x + y*y + z*z) + 0.000001
    return jnp.array([jnp.log(norm), (x/normXYZ)*jnp.arccos(w/norm), (y/normXYZ)*jnp.arccos(w/norm), (z/normXYZ)*jnp.arccos(w/norm)]).T  # Transpose back for the correct shape

@jax.jit
def motion_model(q, tau, omega):
    # qt+1 = f(qt, τtωt) := qt ◦ exp([0, τtωt/2]).
    w, x, y, z = q
    wx, wy, wz = omega
    qr = jnp.array([0.0 * tau , tau*wx/2, tau*wy/2, tau*wz/2])
    exp_qr = quaternion_exp(qr)
    exp_qr1 = jnp.squeeze(exp_qr)
    return quaternion_multiply(q,exp_qr1)

@jax.jit
def observation_model(q):
    # at = h(qt) := q−1t ◦ [0, 0, 0,−g] ◦ qt
    w, x, y, z = q
    qa = jnp.array([0.0 * w , 0.0 * w, 0.0 * w, (0.0 * w) + 1])
    return quaternion_multiply(quaternion_inverse(q) ,quaternion_multiply(qa,q))

@jax.jit
def quaternion_norm2(q):
    w, x, y, z = q.T  
    norm2 = w*w + x*x + y*y + z*z + 0.000001
    return norm2

@jax.jit
def quaternion_normalize(q):
    w, x, y, z = q.T  
    norm = jnp.sqrt(w*w + x*x + y*y + z*z) + 0.000001
    return jnp.array([w/norm, x/norm, y/norm, z/norm]).T

@jax.jit
def cost_fun(qtp1, qt, tau, omega, at):
    # ∥2 log(q−1t+1 ◦ f(qt, τtωt))∥22
    # ∥at − h(qt)∥22
    ax, ay, az = at.T
    a_t = jnp.array([0 * ax, ax, ay, az]).T
    h_qt = jax.vmap(observation_model, in_axes=0)(qt)
    obj1 = jnp.sum(quaternion_norm2(a_t - h_qt))
    qi_tp1 = quaternion_inverse(qtp1) # make it t+1 by shifting down
    f_q = jax.vmap(motion_model, in_axes=(0, 0, 0))(qt, tau, omega)
    obj2 = jnp.sum(quaternion_norm2(2*quaternion_log(quaternion_multiply(qi_tp1, f_q))))
    cost = 0.5*(obj1 + obj2)
    return cost


tau_t = jnp.transpose(jnp.diff(ts))
tau_t = jnp.concatenate([tau_t, jnp.array([[0.0]])], axis=0)
omega_t = jnp.transpose(imu_vals[3:6, :])
acc_t = jnp.transpose(imu_vals[0:3, :])
num_quaternions = omega_t.shape[0]
quaternion_values = jnp.array([1.0, 0.0, 0.0, 0.0])
quaternions = jnp.tile(quaternion_values, (num_quaternions, 1))

# motion model initialization
def update_quaternion(carry, inputs):
    quaternions, tau_t, omega_t = carry
    i, _ = inputs
    updated_q = motion_model(quaternions[i, :], tau_t[i, :], omega_t[i, :])
    return (quaternions.at[i + 1, :].set(updated_q), tau_t, omega_t), None

# Use jax.lax.scan for sequential updates
final_state, _ = jax.lax.scan(update_quaternion, (quaternions, tau_t, omega_t), (jnp.arange(1, num_quaternions - 1), None))
# Extract the final updated quaternions
quaternions_motion = final_state[0]
quaternions = quaternions_motion

max_iter = 1000
alpha = jnp.array([0.001, 0.001, 0.001, 0.001])
q_initial = jnp.array([1.0, 0.0, 0.0, 0.0])
prev_cost = 10000
cost_values = []
for i in range(max_iter):
    cost = cost_fun(quaternions[1:],quaternions[0:-1], tau_t[0:-1], omega_t[0:-1], acc_t[0:-1])
    print("Iter ", i, " Cost:",cost)
    gradient = jax.jacrev(cost_fun,argnums=1)(quaternions[1:],quaternions[0:-1], tau_t[0:-1], omega_t[0:-1], acc_t[0:-1])
    # print(gradient)
    quaternions = quaternions[1:] - alpha*gradient
    quaternions = jnp.concatenate([q_initial[jnp.newaxis, :], quaternions], axis=0)
    quaternions = jax.vmap(quaternion_normalize, in_axes=0)(quaternions)
    if prev_cost<cost or (prev_cost - cost) < 0.0001:
      break
    prev_cost = cost
    cost_values.append(cost)

# plt.plot(range(0, len(cost_values)), cost_values)
# plt.title('Cost vs Iterations')
# plt.xlabel('Iterations')
# plt.ylabel('Cost')
# plt.grid(True)
# plt.show()

def quaternion_to_euler(q):
    w, x, y, z = q
    yaw = jnp.arctan2(2 * (w*x + y*z), 1 - 2 * (x**2 + y**2))
    pitch = jnp.arcsin(2 * (w*y - z*x))
    roll = jnp.arctan2(2 * (w*z + x*y), 1 - 2 * (y**2 + z**2))

    return yaw, pitch, roll

def rotation_matrix_to_euler(rotation_matrix):
    r00, r01, r02 = rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2]
    r10, r11, r12 = rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2]
    r20, r21, r22 = rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2]
    roll = jnp.arctan2(r01, r00)
    pitch = jnp.arctan2(-r02, jnp.sqrt(r12**2 + r22**2))
    yaw = jnp.arctan2(r12, r22)
    return roll, pitch, yaw

roll_imu, pitch_imu, yaw_imu = jax.vmap(quaternion_to_euler, in_axes=0)(quaternions)
# At = At[:yaw_values.shape[0]]
Bt = Bt.flatten()
# At = At.flatten()

# Plot yaw, pitch, and roll against time
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(Bt, yaw_imu, label='Estimated Yaw')
plt.title('Yaw Values')
plt.xlabel('Time')
plt.ylabel('Yaw (radians)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(Bt, pitch_imu, label='Estimated Pitch')
plt.title('Pitch Values')
plt.xlabel('Time')
plt.ylabel('Pitch (radians)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(Bt, roll_imu, label='Estimated Roll')
plt.title('Roll Values')
plt.xlabel('Time')
plt.ylabel('Roll (radians)')
plt.legend()

plt.tight_layout()
plt.show()
