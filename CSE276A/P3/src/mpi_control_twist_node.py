#!/usr/bin/env python
""" MegaPi Controller ROS Wrapper"""
import rospy

from geometry_msgs.msg import Twist
from mpi_control import MegaPiController
import numpy as np
import time

class MegaPiControllerNode:
    def __init__(self, verbose=False, debug=False):
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=verbose)
        self.r = 0.025 # radius of the wheel
        self.lx = 0.055 # half of the distance between front wheel and back wheel
        self.ly = 0.07 # half of the distance between left wheel and right wheel
        self.calibration = 100.0

    def twist_callback(self, twist_cmd):
        desired_twist = self.calibration * np.array([[twist_cmd.linear.x], [twist_cmd.linear.y], [twist_cmd.angular.z]])
        # calculate the jacobian matrix
        jacobian_matrix = np.array([[1, -1, -(self.lx + self.ly)],
                                     [1, 1, (self.lx + self.ly)],
                                     [1, 1, -(self.lx + self.ly)],
                                     [1, -1, (self.lx + self.ly)]]) / self.r
        # calculate the desired wheel velocity
        result = np.dot(jacobian_matrix, desired_twist)
        print(result[0][0], result[1][0], result[2][0], result[3][0])
        # result = np.clip(result, -100, 100)
        self.mpi_ctrl.setFourMotors(result[0][0], result[1][0], result[2][0], result[3][0])

    def run(self):
        time.sleep(1)
        self.mpi_ctrl.setFourMotors(100, 30, 100, 30)
        time.sleep(10)
        self.mpi_ctrl.setFourMotors(0, 0, 0, 0)

if __name__ == "__main__":
    rospy.init_node('megapi_controller')
    mpi_ctrl_node = MegaPiControllerNode(verbose=True)
    # mpi_ctrl_node.run()
    rospy.Subscriber('/twist', Twist, mpi_ctrl_node.twist_callback, queue_size=1) 
    rospy.spin()
