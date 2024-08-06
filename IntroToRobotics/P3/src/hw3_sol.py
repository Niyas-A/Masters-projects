#!/usr/bin/env python
import time
import rospy
import sys
import rospy
from geometry_msgs.msg import Twist
import numpy as np
from pid_controller import PIDcontroller, genTwistMsg, coord
from geometry_msgs.msg import TransformStamped
from april_detection.msg import AprilTagDetectionArray
import numpy as np
import math
import tf
from geometry_msgs.msg import TransformStamped
import time
from threading import Thread, Lock
import os

kalman_mutex = Lock()

class Logging:
    def __init__(self, max_size=8, total_state_size=19):
        self.file = "report"
        self.landmark_file = self.get_unique_filename("landmark_" + self.file, "csv")
        self.x_pos = self.get_unique_filename("x_" + self.file, "csv")
        self.states = self.get_unique_filename("states_" + self.file, "csv")

        with open(self.x_pos, "a+") as f:
            f.write("time,x,y,theta,v_x,v_y,v_theta" + "\n")
        with open(self.landmark_file, "a+") as f:
            f.write("time,id,x,y,yaw,robot_x,robot_y,robot_theta" + "\n")

        with open(self.states, "a+") as f:
            headers = []
            headers.append("robot_x")
            headers.append("robot_y")
            headers.append("robot_theta")
            for i in range(max_size):
                headers.append("x_" + str(i))
                headers.append("y_" + str(i))
                # headers.append("theta_" + str(i))
            f.write("time," + ",".join(headers) + "\n")
        
        self.total_state_size = total_state_size
        self.max_size = max_size

        self.start_time = time.time()

    def get_unique_filename(self, base_filename, extension):
        counter = 0
        while True:
            filename = "{}_{}.{}".format(base_filename, counter, extension)
            if not os.path.exists(filename):
                print("Saving to file", filename)
                return filename
            counter += 1

    def write_x(self, x, v):
        current_time = time.time() - self.start_time
        state = [current_time, x[0], x[1], x[2], v[0], v[1], v[2]]
        with open(self.x_pos, "a+") as f:
            f.write(self.write_str_list(state) + "\n")
    
    def write_str_list(self, lst):
        return ",".join([item if isinstance(item, str) else str(np.round(item, 3)) for item in lst])

    def write_landmark(self, measurements, current_state):
        robot_x, robot_y, robot_theta = (
            current_state[0],
            current_state[1],
            current_state[2],
        )
        current_time = time.time() - self.start_time
        with open(self.landmark_file, "a+") as f:
            for id, z_i in measurements:
                x, y, yaw = z_i[0], z_i[1], z_i[2]
                state = [current_time, id, x, y, np.round((yaw * 180/np.pi), 2), robot_x, robot_y, robot_theta]
                f.write(self.write_str_list(state) + "\n")

    def write_states(self, x):
        current_time = time.time() - self.start_time
        x_log = x.copy().tolist()
        x_log += [''] * (self.total_state_size - len(x))
        x_log = [current_time] + x_log
        with open(self.states, "a+") as f:
            f.write(self.write_str_list(x_log) + "\n")

class Hw3Node:
    dt = 0.05

    # robot q > landmark q since more noise
    # ROBOT_Q = 0.0025 ** 2

    ROBOT_Q = 0.025 ** 2 / 10
    LANDMARK_Q = 0.01 ** 2

    LANDMARK_R = 0.2 ** 2  # R much smaller than Q larger R
# 
    # More certain about robot inital position robot p lower than landmark p
    ROBOT_P = 0.00000001 # reducing both of them
    LANDMARK_P = 1000 # reducing both of them

    NUM_KINEMATIC = 3
    NUM_VARS_PER_LANDMARK = 2

    def __init__(self, verbose=True):
        self.pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)
        self.april_tag_location = {}
        self.pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)
        self.detection_positon = None
        self.verbose = verbose
        self.time_stamp = None
        self.current_state = (0.0, 0.0, 0.0)
        self.id_detected = []

        self.x = np.array([0.0, 0.0, 0])
        self.F = np.eye(self.NUM_KINEMATIC)
        self.G = self.get_G()
        self.Q = np.eye(self.NUM_KINEMATIC) * self.ROBOT_Q
        self.P = np.eye(self.NUM_KINEMATIC) * self.ROBOT_P  # initial value of P low

        self.logging = Logging()
        self.valid_landmarks = list(range(0, 8))
    
    def get_G(self):
        num_columns = self.state_size - self.NUM_KINEMATIC
        original_array = np.eye(self.NUM_KINEMATIC) * 1
        zeros_columns = np.zeros((original_array.shape[0], num_columns))
        new_array = np.hstack((original_array, zeros_columns))
        return new_array.T

    @property
    def state_size(self):
        return self.NUM_KINEMATIC + self.NUM_VARS_PER_LANDMARK * len(self.id_detected)

    def predict_kalman(self, u):
        with kalman_mutex:
            self.x = np.dot(self.F, self.x) + np.dot(self.G, u)
            self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
            self.logging.write_x(self.x, u)
            self.logging.write_states(self.x)

    def handle_new_measurements(self, measurements):
        unknown_measures = [
           (z_id, z_i) for (z_id, z_i) in measurements if z_id not in self.id_detected
        ]
        if len(unknown_measures) == 0:  # no unknown measurements so no resize
            return 0
        old_x = self.x.copy()
        for z_id, z_i in unknown_measures:
            z_x, z_y, z_yaw = z_i
            old_state_size = self.state_size  # get old state size before adding id
            self.id_detected.append(z_id)
            print("state_size", self.state_size, self.id_detected)
            # update old_x
            new_P = np.eye(self.state_size) * self.LANDMARK_P
            new_P[:old_state_size, :old_state_size] = self.P
            self.P = new_P

            old_x = np.append(old_x, [0, 0]) # TODO
            self.x = old_x

        # Update Q, F, G
        self.F = np.eye(self.state_size)
        self.Q = np.eye(self.state_size) * (self.LANDMARK_Q ** 2)
        self.Q[: self.NUM_KINEMATIC, : self.NUM_KINEMATIC] = np.eye(3) * self.ROBOT_Q
        self.G = self.get_G()

    def get_robot_theta(self):
        return self.x[2]  # 3rd index is always theta

    def get_robot_frame_rotation(
        self, num_measurements
    ):  # Creates num_measurements x num_measurement matrix
        theta = -self.get_robot_theta()
        R_i = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        num_blocks = num_measurements // self.NUM_VARS_PER_LANDMARK
        result_matrix = np.kron(np.eye(num_blocks), R_i)
        return result_matrix

    def update_kalman(self, measurement):
        with kalman_mutex:
            self.handle_new_measurements(measurement)
            self.logging.write_landmark(measurement, self.x) # Logging

            num_measurements = len(measurement) * self.NUM_VARS_PER_LANDMARK
            if num_measurements == 0:
                return
            H = []
            z = []
            self.R = np.eye(num_measurements) * self.LANDMARK_R
            for (id, z_i) in measurement:
                id_index = self.id_detected.index(id)
                for j in range(self.NUM_VARS_PER_LANDMARK):
                    row = [0 for _ in range(self.state_size)]
                    row[j] = -1
                    row[self.NUM_KINEMATIC + id_index * self.NUM_VARS_PER_LANDMARK + j] = 1
                    H.append(row)
                z_i_x, z_i_y, z_i_yaw = z_i
                z.append(z_i_x)
                z.append(z_i_y)
                # z.append(z_i_yaw)
            # print(np.array(H))
            R = self.get_robot_frame_rotation(num_measurements)
            
            # Add rotation to H
            H = np.dot(R, np.array(H))
            z = np.array(z)
    
            # Residual covariance
            S = np.dot(H, np.dot(self.P, H.T)) + self.R

            # Kalman gain
            K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))

            # Measurement residual
            y = z.T - np.dot(H, self.x)
            # Update state estimate
            self.x = self.x + np.dot(K, y)

            # Identity matrix
            I = np.eye(self.F.shape[0])

            # Update error covariance matrix
            self.P = np.dot((I - np.dot(K, H)), self.P)
            # print(self.P.round(2))
    def adjust_to_2pi(self, value):
        adjusted_value = (value + 2 * np.pi) % (2 * np.pi)
        return adjusted_value

    def get_xy_yaw_from_pose(self, detection_id, pose):
        # Extract x, y from the transformed position
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        # Extract yaw (rotation around the z-axis) from the transformed orientation
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        _, yaw, _ = tf.transformations.euler_from_quaternion(quaternion)
        # print(detection_id, np.array(tf.transformations.euler_from_quaternion(quaternion)))
        # TODO check originl was _, yaw, _
        return z, -x, self.adjust_to_2pi(-yaw)

    def update_camera_callback(self, tf_msg):
        measurements = []
        for detection in tf_msg.detections:
            detection_id = detection.id
            if detection_id not in self.valid_landmarks:
                continue
            pose = detection.pose
            x, y, yaw = self.get_xy_yaw_from_pose(detection_id, pose)
            # print(detection_id, x, y, yaw * 180/np.pi)
            measurements.append([detection_id, (x, y, yaw)])
        self.update_kalman(measurements)

    def run(self, enable_slam=False):
        time.sleep(3)
        self.time_stamp = time.time()

        waypoint = np.array(
            [
                [0.0, 0.0, 0.0],
                [.5, 0, 0],
                [0.5, 0.0, np.pi/2],
                [0.5, 0.5, np.pi/2],
                [0.5, 0.5, np.pi],
                [0, 0.5, np.pi],
                [0, 0.5, np.pi * 3/2],
                [0, 0, np.pi * 3/2],
                [0, 0, 0],
            ]
        )
        waypoint[:,0] *= .8
        waypoint[:,1] *= .8
        # init pid controller
        # pid = PIDcontroller(0.010, 0.0015, 0.004)
        pid = PIDcontroller(0.04, 0.0015, 0.004)
        # init current state
        current_state = np.array([0.0, 0.0, 0.0])
        start_time = time.time()
        # in this loop we will go through each way point.
        # once error between the current state and the current way point is small enough,
        # the current way point will be updated with a new point.
        for wp in waypoint:
            print("move to way point", wp)
            # set wp as the target point
            pid.setTarget(wp)

            # calculate the current twist
            update_value = pid.update(current_state)
            # publish the twist
            self.pub_twist.publish(genTwistMsg(coord(update_value, current_state)))
            # print(coord(update_value, current_state))
            time.sleep(0.05)
            # update the current state
            current_state += update_value * 1
            u = np.array([update_value[0], update_value[1], update_value[2]])
            self.predict_kalman(u)
            while (
                np.linalg.norm(pid.getError(current_state, wp)) > 0.05
            ):  # check the error between current state and current way point
                # calculate the current twist
                update_value = pid.update(current_state)
                # publish the twist
                twist = genTwistMsg(coord(update_value, current_state))
                self.pub_twist.publish(twist)
                # print(coord(update_value, current_state))
                time.sleep(0.05)
                # update the current state
                current_state += update_value * 1  # .7
                self.current_state = current_state

                u = np.array([update_value[0], update_value[1], update_value[2]])
                self.predict_kalman(u)
                if enable_slam:
                    current_state = np.array([self.x[0], self.x[1], self.x[2]])
            # time.sleep(1)
        # stop the car and exit
        self.pub_twist.publish(genTwistMsg(np.array([0.0, 0.0, 0.0])))
        print("Completed")
        print(np.diag(self.P))

def main():
    rospy.init_node("hw3")
    hw3_node = Hw3Node()
    rospy.Subscriber(
        "/apriltag_detection_array",
        AprilTagDetectionArray,
        hw3_node.update_camera_callback,
    )
    hw3_node.run()
    hw3_node.run(enable_slam=True)
    hw3_node.run(enable_slam=True)
    # hw3_node.run()
    rospy.spin()

if __name__ == "__main__":
    main()
