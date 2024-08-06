import numpy as np
p = [[0.584538310148106, 0.3969216895055261, -0.7563991718508761, 1.1123947814880453],
 [0.8351703498982022, -0.29383180203223486, 0.5313181769127384, -0.4817367548218464],
 [-0.009434000891466704, -0.8905235566672495, -0.3561759661778382, 0.21178737176561227],
 [0,0,0,1]]
# print(np.linalg.inv(p))
# cam2arm
c2Tw = np.array([[-0.068,0.352,-0.948,1.247],[1.015,-0.013,0.069,-0.177],[-0.041,-0.951,-0.291,0.217],[0,0,0,1]])
# cam2arm_ros
yaw = np.pi
pitch = np.pi/6
Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
c1Tw = np.eye(4)
c1Tw[:3,:3] = Rz@Ry
c1Tw[:3,3] = [1.5,0,0.3]
# c1Tw = c1Tc2@c2Tw
# c1Tw is wTc1 - in ros
c1Tc2 = c1Tw@np.linalg.inv(c2Tw)
print(c1Tc2)
