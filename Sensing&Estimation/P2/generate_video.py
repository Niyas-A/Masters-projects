import cv2
import os
def images_to_video(image_folder, output_video, fps=30):
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    # image_files.sort()  # Sort the files numerically/alphabetically

    frame = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = frame.shape

    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    n = len(image_files)
    # dataset = 20
    i = 0
    for i in range(1,n+1):
        if i%100 == 0:
            print(i)
            # break
        img_path = os.path.join(image_folder, 'rgb20_' + str(i) + '.png')
        frame = cv2.imread(img_path)
        # frame = cv2.imread(img_path)
        video_writer.write(frame)
        # i +=1
    video_writer.release()


image_folder = '../data/RGB20/'
output_video = '../Final Output/dataset20_rgb.mp4'
fps=60
images_to_video(image_folder, output_video, fps)


# import matplotlib.pyplot as plt
# import numpy as np
# import imageio
#
# # Generate some sample data
# x = np.linspace(0, 2*np.pi, 100)
# iterations = 10  # Number of iteratix`ons
#
# # Generate plots for each iteration and save as frames
# frames = []
# for i in range(iterations):
#     y = np.sin(x + i * np.pi / 10)
#     plt.plot(x, y)
#     plt.title(f"Iteration {i}")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.grid(True)
#     # Save plot as an image
#     filename = f"frame_{i:03d}.png"
#     plt.savefig(filename)
#     plt.close()
#     frames.append(filename)
#
# # Compile frames into a video
# with imageio.get_writer('output_video.mp4', mode='I') as writer:
#     for frame in frames:
#         image = imageio.imread(frame)
#         writer.append_data(image)
