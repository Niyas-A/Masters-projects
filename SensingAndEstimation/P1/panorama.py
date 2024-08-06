import pickle
import sys
import time
from math import pi
import numpy as np
import cv2
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

dataset="2"
cfile = "../data/cam/cam" + dataset + ".p"
ifile = "../data/imu/imuRaw" + dataset + ".p"
vfile = "../data/vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import math

A = vicd['rots']
At = vicd['ts']
C = camd['cam']
Ct = camd['ts']
print(C.shape)

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
    index = np.argmin(np.abs(At - t))
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
    R = A[:,:,index] # take inverse
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