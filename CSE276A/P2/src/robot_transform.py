#!/usr/bin/env python
import rospy
import tf2_ros
import geometry_msgs.msg

rospy.init_node("hw2")

robot_to_marker1 = geometry_msgs.msg.TransformStamped()
robot_to_marker1.header.stamp = rospy.Time.now()
robot_to_marker1.header.frame_id = "robot_frame"  # Replace with the actual robot frame
robot_to_marker1.child_frame_id = "marker1_frame"  # Replace with the actual marker1 frame

# Set the translation (position) and rotation (orientation)
robot_to_marker1.transform.translation.x = 0.012600956031
robot_to_marker1.transform.translation.y = -0.126054129912
robot_to_marker1.transform.translation.z = 0.452546189741
robot_to_marker1.transform.rotation.x = 0.594743428128
robot_to_marker1.transform.rotation.y = 0.014706366
robot_to_marker1.transform.rotation.z = 0.0155680816582
robot_to_marker1.transform.rotation.w = 0.803630270915

# Publish the transformation from robot frame to marker1 frame
tf_broadcaster = tf2_ros.TransformBroadcaster()
tf_broadcaster.sendTransform(robot_to_marker1)

rospy.spin()
