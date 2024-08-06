########
# Author: Anand Kumar, Ben Klingensmith, Niyas Attasseri
# Name: saveImg.py
#
# Purpose: RobThePupper. Script to save the image repeatedly from the OAK-D camera.
#
# Usage: This has to launched after the running the camera node to save the image.
#        ros2 launch depthai_ros_driver camera.launch.py
#        ros2 run robthepupper saveImg
#
# Date: 12 June 2024
########

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import time


#####
# Name: ImageSaverNode
# Purpose: This will save the image from the OAK-D camera with a delay of 1 second.
#####
#####
class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver')
        # Subscribe to the camera image topic
        self.subscription = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.image_count = 0
        self.current_time = time.time()


    ###
    # Name: listener_callback
    # Purpose: This will save the image from the OAK-D camera.
    # Arguments: self, the message from the camera
    ###
    def listener_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Save the image
            img_filename = f'/home/ubuntu/transfer_dir/camera_image.png'
            if self.current_time - time.time() > 1:           
                cv2.imwrite(img_filename, cv_image)
                self.current_time = time.time()
                self.get_logger().info(f'Saved image {img_filename}')
            self.image_count += 1
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')
####
# Name: Main
# Purpose: Main function to run the node which saves the image from the OAK-D camera.
# #####       
def main(args=None):
    rclpy.init(args=args)
    image_saver_node = ImageSaverNode()
    try:
        rclpy.spin(image_saver_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        image_saver_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
