########
# Author: Anand Kumar, Ben Klingensmith, Niyas Attasseri
# Name: gui_pupper_service_ssh
#
# Purpose: RobThePupper. Service for the pupper to move based on the key.txt file sent from the laptop.
#
# Usage: This has to launched before the client for the movement commands to be published.
#        ros2 run robthepupper gui_service
#
# Date: 12 June 2024
########

# Import the ROS2 interface we wrote, called GoPupper. This specifies the message type.
from pupper_interfaces.srv import GoPupper

# packages to let us create nodes and spin them up
import rclpy
from rclpy.node import Node

# The Twist package is what we use to move the robot
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose

# To let us set timers
import time   # Is on our side, yes it is. And if you don't know the song, here you go! https://www.youtube.com/watch?v=8wDUhw15E-s

####
# Name: Minimal Service
#
# Purpose: "The MinimalService class constructor initializes the node with the name minimal_service. 
# Then, it creates a service and defines the type, name, and callback.""
#
#
####
class MinimalService(Node):

    # Constructor
    def __init__(self):
        # Initialize the node 
        super().__init__('minimal_service')

        # Create the service, defining its type (GoPupper), name (pup_command), and callback.
        self.srv = self.create_service(GoPupper, 'pup_command', self.pup_callback)
        
        # publish twist 
        self.vel_publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # publish orientation
        self.orientation_publisher_ = self.create_publisher(Pose, 'reference_body_pose', 10)

        # timer interval (need to wait between messages)
        self.interval = 0.5  # .5 seconds


    #####
    # Name: pup_callback
    # Purpose: This will receive pupper movement messages and publish accordingly.
    # Arguments: self, the request (e.g., the command from client), the response (e.g., success)
    #####
    def pup_callback(self, request, response):
        # We'll be publishing a velocity message. Calling the Twist constructor zeroes it out.  
        velocity_cmd = Twist()
        orientation_cmd = Pose()
        ## Debug - if you're curious what message this method got, uncomment this out
        #print("In server pup_callback, got this command: %s" % request.command)

        ## Here is a set of conditionals - move forward, move_backward, etc, and they send their
        # respective linear velocity commands accordingly. See Lab 0 / Lab 1 to learn more about this. 
        if (request.command == 'move_forward'):
            velocity_cmd.linear.x = 0.5    # .5 in the linear X direction moves us forward
            self.vel_publisher_.publish(velocity_cmd)   # publish the command
            self.get_logger().info('Publishing: "%s"' % request.command)  # Log what happened
            time.sleep(self.interval)  # Wait and make sure the robot moved

        elif (request.command == 'move_backward'):
            velocity_cmd.linear.x = -0.5
            self.vel_publisher_.publish(velocity_cmd)
            self.get_logger().info('Publishing: "%s"' % request.command)
            time.sleep(self.interval)   

        elif (request.command == 'move_left'):
            velocity_cmd.linear.y = 0.5
            self.vel_publisher_.publish(velocity_cmd)
            self.get_logger().info('Publishing: "%s"' % request.command)
            time.sleep(self.interval)   

        elif (request.command == 'move_right'):
            velocity_cmd.linear.y = -0.5
            self.vel_publisher_.publish(velocity_cmd)
            self.get_logger().info('Publishing: "%s"' % request.command)
            time.sleep(self.interval)   

        elif (request.command == 'turn_left'):
            velocity_cmd.angular.z = 1.0
            self.vel_publisher_.publish(velocity_cmd)
            self.get_logger().info('Publishing: "%s"' % request.command)
            time.sleep(self.interval)  

        elif (request.command == 'look_down'):
            orientation_cmd.orientation.y = 0.149
            orientation_cmd.orientation.w = 0.989
            self.orientation_publisher_.publish(orientation_cmd)
            self.get_logger().info('Publishing: "%s"' % request.command)
            time.sleep(self.interval)

        elif (request.command == 'look_up'):
            orientation_cmd.orientation.y = -0.149
            orientation_cmd.orientation.w = 0.989
            self.orientation_publisher_.publish(orientation_cmd)
            self.get_logger().info('Publishing: "%s"' % request.command)
            time.sleep(self.interval)

        elif (request.command == 'look_straight'):
            orientation_cmd.orientation.y = 0.0
            orientation_cmd.orientation.w = 1.0
            self.orientation_publisher_.publish(orientation_cmd)
            self.get_logger().info('Publishing: "%s"' % request.command)
            time.sleep(self.interval)

        elif (request.command == 'turn_right'):
            velocity_cmd.angular.z = -1.0
            self.vel_publisher_.publish(velocity_cmd)
            self.get_logger().info('Publishing: "%s"' % request.command)
            time.sleep(self.interval)


        elif (request.command == 'stay'):
            time.sleep(self.interval)  # do nothing

        else:
            self.get_logger().info('Invalid command: "%s"' % request.command)
            time.sleep(self.interval)  # do nothing

        # Stop the robot from moving (set everyting to zero by calling the Twist constructor)
        velocity_cmd = Twist()
        self.vel_publisher_.publish(velocity_cmd)

        # Give a response. (Probably we should set this to false if the command was invalid per above logic,
        # but this is just demo code to give you an idea).  
        response.executed = True
        return response

####
# Name: Main
# Purpose: Main functoin to set up our service
#####
def main():
    # Initialize the python client library in ROS 2
    rclpy.init()

    # Instatiate the class & create the node for the service
    minimal_service = MinimalService()

    # Spin the node - this will handle call backs
    rclpy.spin(minimal_service)

    # Destroy the node when we're done with it
    minimal_service.destroy_node()
    
    # Shutdown  
    rclpy.shutdown()


if __name__ == '__main__':
    main()