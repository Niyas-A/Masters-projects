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
from transformation_camera_to_world import CameraTransformationHandler
import numpy as np
import math
import tf
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_inverse, quaternion_from_euler

class Hw2Node:
    def __init__(self, verbose=True):
        self.pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)

        self.waypoint = np.array([[0.0,0.0,0.0], 
                        [1.0,0.0,0.0],
                        [1.0,2.0,np.pi],
                        [0, 0, 0, 0]]) 
        self.transform_handler = CameraTransformationHandler()

        self.april_tag_location = {
            3: self.transform_handler.create_transformation_matrix(0, 0, 1, 0 ,0, 0),
        }

        self.detection_positon = None
        self.verbose = verbose
        
        self.tf_broadcast = tf.TransformBroadcaster()
        
    def update_camera_callback(self, tf_msg):
        # self.tf_broadcast.sendTransform((0, 0, 0), (0, 0, 0, 0), rospy.Time.now(), "world_frame", "robot")
        avg_x, avg_y, avg_angle = 0, 0, 0
        for detection in tf_msg.detections:
            print(detection)
            detection_id = detection.id
            pos = detection.pose.position
            ori = detection.pose.orientation
            detection_pose = (-pos.x, -pos.y, -pos.z)
            inverse_rotation = quaternion_inverse((ori.x, ori.y, ori.z, ori.w))

            april_tag_transformation_matrix = self.april_tag_location[detection_id]
            x, y, angle = self.transform_handler.get_robot_position(
                detection,
                april_tag_transformation_matrix
            )
            avg_x += x
            avg_y += y
            avg_angle += angle

        if tf_msg.detections:
            avg_x /= len(tf_msg.detections)
            avg_y /= len(tf_msg.detections)
            avg_angle /= len(tf_msg.detections)
            self.detection_positon = (avg_x, avg_y, avg_angle)
            if self.verbose:
                print("Detection postion", self.detection_positon)
        else:
            self.detection_positon = None

    def run(self):
            # init pid controller
        pid = PIDcontroller(0.02,0.005,0.005)
        pid.timestep = .05

        current_state = np.array([0.0,0.0,0.0])
        for wp in self.waypoint:
            print("move to way point", wp)
            # set wp as the target point
            current_state = self.detection_positon            
            pid.setTarget(wp)
            while(np.linalg.norm(pid.getError(current_state, wp)) > 0.05): # check the error between current state and current way point
                # calculate the current twist

                update_value = pid.update(current_state)
                # publish the twist
                self.pub_twist.publish(genTwistMsg(coord(update_value, current_state)))
                #print(coord(update_value, current_state))
                time.sleep(0.05)
                # update the current state
                if self.detection_positon:
                    current_state += self.detection_positon
                else:
                    current_state += update_value

        # stop the car and exit
        self.pub_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))


def main():
    rospy.init_node("hw2")
    hw2_node = Hw2Node()
    rospy.Subscriber('/apriltag_detection_array', AprilTagDetectionArray, hw2_node.update_camera_callback)
    rospy.spin()

    # Test camera detection position and calibration
        # Verify the x, y, angle is correct
    # Test multiple camera detection
    # Test movement basic
    # 

if __name__ == "__main__":
    main()
