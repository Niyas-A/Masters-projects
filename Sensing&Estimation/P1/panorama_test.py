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
from math import pi
import cv2
from transforms3d.quaternions import quat2mat

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

dataset="11"
cfile = "../testset/cam/cam" + dataset + ".p"
ifile = "../testset/imu/imuRaw" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
toc(ts,"Data import")

B = imud['vals']
Bt = imud['ts']
# A = vicd['rots']
# At = vicd['ts']
C = camd['cam']
Ct = camd['ts']
print(C.shape)
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

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

dx = 45*pi/(240*180)
dy = 60*pi/(320*180)
length = 2000
bredth = 1000
img = np.zeros((length,bredth,3), dtype=np.uint8)

#C.shape[3]
for k in range(C.shape[3]):
    if(k%100==0):
        print(k)
    image = C[:,:,:,k]
    t = Ct[:,k]
    index = np.argmin(np.abs(Bt - t))
    data = []
    l, b, c = image.shape
    for i in range(l):
      for j in range(b):
        lat = (i - l/2)*dx
        lon = (j - b/2)*dy
        data.append([lat,lon,1,image[i,j,0],image[i,j,1],image[i,j,2]])
    data = np.array(data)
    cartesian_coordinates = sph2cart(data[:, 1], data[:, 0], data[:, 2])
    cartesian_coordinates = np.array(cartesian_coordinates).T
    R = quat2mat(quaternions[index,:])
    world_cartesian_coordinates = R@cartesian_coordinates.T
    x, y, z = world_cartesian_coordinates
    az, el, r_w = cart2sph(x, y, z)
    image_index_x = az*length/(2*pi) + length/2
    image_index_y = el*bredth/(2*pi) + bredth/2
    for i in range(len(image_index_x)):
      img[int(image_index_x[i]),int(image_index_y[i])] = data[i,3:]
# cv2.imshow(img)
rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title('Panorama Image')
plt.axis('off')  # Hide axis
plt.show()