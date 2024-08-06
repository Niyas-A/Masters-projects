import numpy as np
from math import cos, sin, radians
import math
from tf.transformations import euler_matrix, translation_matrix, quaternion_matrix, concatenate_matrices, inverse_matrix, translation_from_matrix

class CameraTransformationHandler:
    def __init__(self):
        """
        # Computing T_camera_to_world
        R, _ = cv2.Rodrigues(rvecs)

        # Step 2: Create T_camera_to_world
        T_camera_to_world = np.identity(4)
        T_camera_to_world[:3, :3] = R  # Set the upper-left 3x3 block to the rotation matrix
        T_camera_to_world[:3, 3] = tvecs.ravel()  # Set the first three elements of the last column to the translation vector
        """
        pass

    def get_robot_position(self, april_tag_detection, T_world_to_april):

        T_camera_space_to_april_tag = self.get_T_camera_space_to_april_tag(april_tag_detection.pose)
        T_april_camera = inverse_matrix(T_camera_space_to_april_tag)
        T_camera_to_robot = tf.matrix([
            [0,,0,],
        ])
        T_world_to_r = T_world_to_april.dot(T_april_camera)
        x, y, z = translation_from_matrix(T_world_to_r)
        return x, y, z,

    def get_T_camera_space_to_april_tag(self, pose):
        tr = translation_matrix((pose.position.x, pose.position.y, pose.position.z))
        qm = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        return concatenate_matrices(tr, qm)

    def extract_x_y_yaw_from_world_space(self, T):
        # Extract the X and Y coordinates from the transformation matrix
        x = T[0, 3]
        y = T[1, 3]
        # Calculate the Yaw (rotation around the Z-axis) using atan2
        yaw = np.arctan2(T[1, 0], T[0, 0])
        return x, y, yaw



    def create_transformation_matrix(self, x, y, z, roll, pitch, yaw):
        R = euler_matrix(roll, pitch, yaw)
        T = translation_matrix((x, y, z))
        # Combine the transformation matrices using matrix multiplication
        combined_matrix = T.dot(R)
        return combined_matrix