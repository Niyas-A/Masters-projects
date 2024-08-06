import numpy as np
import matplotlib.pyplot as plt
import os

frame_interval = 100  # Interval at which to save frames
output_folder = "frames"  # Folder to save the frames

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


dataset = 21

with np.load("../data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

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

for frame in range(0, pose_imu.shape[1], frame_interval):
    plt.figure(figsize=(10, 6))
    plt.plot(pose_imu[0, :frame+1], pose_imu[1, :frame+1], label='IMU & Encoder Odometry')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Robot position over time')
    plt.grid(True)
    plt.legend()
    # Save the frame
    plt.savefig(f"{output_folder}/frame_{frame:04d}.png")
    plt.close()

import cv2
import os
import numpy as np

frame_folder = 'frames'
output_video = 'trajectory_video.mp4'
fps = 10  # Frames per second in the output video

images = [img for img in os.listdir(frame_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(frame_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

for image in sorted(images):
    video.write(cv2.imread(os.path.join(frame_folder, image)))

cv2.destroyAllWindows()
video.release()
