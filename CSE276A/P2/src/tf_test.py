#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler, quaternion_conjugate, quaternion_inverse
from april_detection.msg import AprilTagDetectionArray
pos_global = (0, 0, 0)
ori_global = (0, 0, 0, 1)
def camera_callback(tf_msg):
    global pos_global, ori_global
    for detection in tf_msg.detections:
        #print(detection)
        detection_id = detection.id
        pos = detection.pose.position
        ori = detection.pose.orientation
        #print(pos)
        #print(ori)
        pos_global = (pos.x, pos.y, pos.z)
        ori_global = (ori.x, ori.y, ori.z, ori.w)
        #detection_pose = (-pos.x, -pos.y, -pos.z)
        #inverse_rotation = quaternion_inverse((ori.x, ori.y, ori.z, ori.w))        


if __name__ == '__main__':
    rospy.init_node('marker_transform_publisher')
    broadcaster = tf2_ros.TransformBroadcaster()
    rospy.Subscriber('/apriltag_detection_array', AprilTagDetectionArray, camera_callback)
    rate = rospy.Rate(10.0)  # 10 Hz
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    while not rospy.is_shutdown():
        current_time = rospy.Time.now()

        # Transformation from marker1_frame to world_frame
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = current_time
        transform_stamped.header.frame_id = "world_frame"
        transform_stamped.child_frame_id = "marker1_frame"
        transform_stamped.transform.translation.x = 0.0
        transform_stamped.transform.translation.y = 0.0
        transform_stamped.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, 0)  # No rotation
        transform_stamped.transform.rotation.x = q[0]
        transform_stamped.transform.rotation.y = q[1]
        transform_stamped.transform.rotation.z = q[2]
        transform_stamped.transform.rotation.w = q[3]
        broadcaster.sendTransform(transform_stamped)

        # Transformation from marker2_frame to world_frame
        transform_stamped.child_frame_id = "marker2_frame"
        transform_stamped.transform.translation.x = 1.0
        transform_stamped.transform.translation.y = 0.0
        transform_stamped.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, 3.14159)  # 180 degrees rotation around Z-axis
        transform_stamped.transform.rotation.x = q[0]
        transform_stamped.transform.rotation.y = q[1]
        transform_stamped.transform.rotation.z = q[2]
        transform_stamped.transform.rotation.w = q[3]
        broadcaster.sendTransform(transform_stamped)

        # Transformation from marker3_frame to world_frame
        transform_stamped.child_frame_id = "marker3_frame"
        transform_stamped.transform.translation.x = 1.0
        transform_stamped.transform.translation.y = 1.0
        transform_stamped.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, 0.7854)  # 45 degrees rotation around Z-axis
        transform_stamped.transform.rotation.x = q[0]
        transform_stamped.transform.rotation.y = q[1]
        transform_stamped.transform.rotation.z = q[2]
        transform_stamped.transform.rotation.w = q[3]
        broadcaster.sendTransform(transform_stamped)

        robot_to_marker1 = TransformStamped()
        robot_to_marker1.header.stamp = rospy.Time.now()
        robot_to_marker1.header.frame_id = "marker2_frame"  # The parent frame is marker1_frame
        robot_to_marker1.child_frame_id = "robot_frame"  # The child frame is robot_frame

        # Set the translation (position)
        #robot_to_marker1.transform.translation.x = pos_global[2]
        #robot_to_marker1.transform.translation.y = -pos_global[0]
        #robot_to_marker1.transform.translation.z = pos_global[1]
        #robot_to_marker1.transform.rotation.x = ori_global[2]
        #robot_to_marker1.transform.rotation.y = -ori_global[0]
        #robot_to_marker1.transform.rotation.z = ori_global[1]
        #robot_to_marker1.transform.rotation.w = ori_global[3]
        # Set the rotation (orientation) and reverse it
        robot_to_marker1.transform.translation.x = pos_global[2]
        robot_to_marker1.transform.translation.y = -pos_global[0]
        robot_to_marker1.transform.translation.z = pos_global[1]
        original_rotation = [ori_global[2], -ori_global[0], ori_global[1], ori_global[3]]
        reversed_rotation = quaternion_conjugate(original_rotation)
        robot_to_marker1.transform.rotation.x = reversed_rotation[0]
        robot_to_marker1.transform.rotation.y = reversed_rotation[1]
        robot_to_marker1.transform.rotation.z = reversed_rotation[2]
        robot_to_marker1.transform.rotation.w = reversed_rotation[3]
        broadcaster.sendTransform(robot_to_marker1)
        
        # Look up the transformation from "world_frame" to "robot_frame"
        try:
            transform = tfBuffer.lookup_transform("world_frame", "robot_frame", rospy.Time(0))
            print("Position (X, Y, Z):", transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z)
            print("Orientation (X, Y, Z, W):", transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass    # Print the position and orientation
        rate.sleep()

    rospy.spin()
