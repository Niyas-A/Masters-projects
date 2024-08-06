#!/usr/bin/env python
"""
Copyright 2023, UC San Diego, Contextual Robotics Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import sys
import rospy
from sensor_msgs.msg import Joy
from key_parser import get_key, save_terminal_settings, restore_terminal_settings
import time
import numpy as np
import math
def points_to_joy(points):
    joy_msg = Joy()
    joy_msg.axes = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]
    joy_msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0]
    joy_msg.axes[:len(points)] = points
    return joy_msg
from pprint import pprint
import math
import csv

def smart_atan(x, y):
    if x == 0:
        if y > 0:
            result = math.pi / 2  # arctan(infinity) is +pi/2
        elif y < 0:
            result = -math.pi / 2  # arctan(-infinity) is -pi/2
        else:
            return 0
    elif y == 0:
        if x > 0 :
            result = math.pi
        elif x < 0:
            result = math.pi
    else:
        result = math.atan(y / x)
    return result

def points_to_joy_motors(vfl, vfr, vbl, vbr):
    return points_to_joy([vfl, vfr, vbl, vbr])

class Hw1Node:
    def convert_w_to_power(self, arr):
        vals = []
        for item in arr:
            if abs(item) < 2.5:
                vals.append(0)
            else:
                conversion = 7.99 * abs(item) + 9.84
                vals.append(conversion * np.sign(item))
        return vals
        # return 7.99 * arr + 9.84
    def get_normalized_theta(self, theta):
        print(theta)
        if theta < 0:
            theta = 2 * math.pi + theta
        return abs(theta)
        

    def compute_timings(self):
        prev_x, prev_y, prev_theta = 0, 0, 0
        with open("/root/rb5_ws/src/rb5_ros/hw1_control/src/waypoints.txt", "r") as f:
            rows = f.readlines()
        results = []
        for i, row in enumerate(rows):
            x, y, theta = row.split(",")
            x, y, theta = float(x), float(y), float(theta)
            dist = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            t2 = smart_atan(x - prev_x, y - prev_y)
            if i==0:
                results.append([None, 2])
                continue
    
            secs_per_radian = 4.95/(2*math.pi)
            pre_rotation_time = abs(t2 - prev_theta) * secs_per_radian 
            movement_time = 10/1.92 * dist
            prev_theta = t2
            final_rotation_time = abs(theta - prev_theta) * secs_per_radian
            
            print(x, y, theta, pre_rotation_time, movement_time, final_rotation_time)
            results.append([True, pre_rotation_time])
            results.append([False, movement_time])
            results.append([True, final_rotation_time])

            prev_x, prev_y, prev_theta = x, y, theta
        return results

    def __init__(self):
        self.pub_hw1 = rospy.Publisher("/hw1", Joy, queue_size=1)
        r = .034
        
        ly = .14
        lx = .113

        self.timings = []
        self.speeds = []
        prev_theta = 0
        for rotates, t in self.compute_timings():
            self.timings.append(t)
            if rotates is None:
                self.speeds.append([0, 0])
            elif rotates:
                self.speeds.append([-50, 50])
            else:
                self.speeds.append([50, 50])
            print(self.speeds[-1], self.timings[-1])

        self.start_time = time.time()


    def run(self):
        time.sleep(1)
        self.reset()
        for speed, t in zip(self.speeds, self.timings): 
            left_motor, right_motor = speed[0], speed[1]
            joy_msg = points_to_joy_motors(vfl=left_motor, vfr=right_motor, vbl=left_motor, vbr=right_motor)
            self.pub_hw1.publish(joy_msg)
            time.sleep(t)
            self.reset()
            time.sleep(1)
        self.reset()



    def reset(self):
        joy_msg = points_to_joy([0, 0, 0, 0])
        self.pub_hw1.publish(joy_msg)

if __name__ == "__main__":
    node = Hw1Node()
    rospy.init_node("hw1_node")
    node.run()
