########
# Author: Anand Kumar, Ben Klingensmith, Niyas Attasseri
# Name: gui_pupper_client_ssh
#
# Purpose: RobThePupper. Client for the pupper to move based on the key.txt file sent from the laptop.
#
# Usage: First launch the bringup command, the gui_service and then run the client:
#        ros2 run robthepupper gui_client_ssh
#
#
# Date: 12 June 2024
########

# Import the ROS2 interface we wrote, called GoPupper. This specifies the message type.
import sounddevice as sd
from pupper_interfaces.srv import GoPupper
import readchar
# Import for the touch sensors
import time
import RPi.GPIO as GPIO
import cv2
import numpy as np
import audio2numpy as a2n


# There are 4 areas for touch actions
# Each GPIO to each touch area
touchPin_Front = 6
touchPin_Left = 3
touchPin_Right = 16
touchPin_Back = 2

# Use GPIO number but not PIN number
GPIO.setmode(GPIO.BCM)

# Set up GPIO numbers to input
GPIO.setup(touchPin_Front, GPIO.IN)
GPIO.setup(touchPin_Left, GPIO.IN)
GPIO.setup(touchPin_Right, GPIO.IN)
GPIO.setup(touchPin_Back, GPIO.IN)

# Packages to let us create nodes and spin them up
import rclpy
from rclpy.node import Node

from MangDang.mini_pupper.display import Display, BehaviorState
from resizeimage import resizeimage  # library for image resizing
from PIL import Image, ImageDraw, ImageFont  # library for image manip.

MAX_WIDTH = 320  # max width of the LCD display

# Get access to the display so we can display things
disp = Display()

# Open the expression images and resize and save them
imgLocs = ["/home/ubuntu/tmp_dir/img_dir/front", "/home/ubuntu/tmp_dir/img_dir/left",
           "/home/ubuntu/tmp_dir/img_dir/right", "/home/ubuntu/tmp_dir/img_dir/stop"]
for path in imgLocs:
    imgLoc = path + '.jpg'
    imgFile = Image.open(imgLoc)

    # Convert to RGBA if needed
    if (imgFile.format == 'PNG'):
        if (imgFile.mode != 'RGBA'):
            imgOld = imgFile.convert("RGBA")
            imgFile = Image.new('RGBA', imgOld.size, (255, 255, 255))

    # We likely also need to resize to the pupper LCD display size (320x240).
    # Note, this is sometimes a little buggy, but you can get the idea.
    width_size = (MAX_WIDTH / float(imgFile.size[0]))
    imgFile = resizeimage.resize_width(imgFile, MAX_WIDTH)

    newFileLoc = path + '_new.jpg'  # rename as you like
    # now output it (super inefficient, but it is what it is)
    imgFile.save(newFileLoc, imgFile.format)


###
# Name: Minimal Client Async
#
# Purpose: "The MinimalClientAsync class constructor initializes the node with the name minimal_client_async. "
#          "The constructor definition creates a client with the same type and name as the service node.
#          The type and name must match for the client and service to be able to communicate."
#
######
class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        # super().__init__('client_go_pupper')
        self.cli = self.create_client(GoPupper, 'pup_command')

        # "The while loop in the constructor checks if a service matching the type and name of the client
        # is available once a second."
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        # "Finally it creates a new request object.""
        self.req = GoPupper.Request()

    ###
    # Name: send_move_request
    # Purpose: send_move_request method, send request and spin until receive response or fail
    # Arguments:  self (reference the current class), move_command (the command we plan to send to the server)
    #####
    def send_move_request(self, move_command):
        self.req = GoPupper.Request()
        self.req.command = move_command
        print("In send_move_request, command is: %s" % self.req.command)
        self.future = self.cli.call_async(self.req)  # send the command to the server
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


###
# Name: Main
# Purpose: Main function to run the client which sends the command to the pupper by reading the key.txt file
#####
def main(args=None):
    global minimal_client
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    key = ''
    old_time = ''
    while rclpy.ok():

        #read lines of file in /home/ubuntu/transfer_dir/key.txt which contains the time and key pressed
        with open('/home/ubuntu/transfer_dir/key.txt', 'r') as file:
            time = file.readline()
            key = file.readline()
        
        #if the time is the same as the old time, continue and don't send the same command again
        if time == old_time:
            continue

        #if the key is not empty, send the command to the pupper where 'w' is forward,
        # 'a' is left, 'd' is right, 's' is backward, 'q' is turn left, 'e' is turn right,
        # 'z' is look up, 'c' is look down, 'x' is look straight
        #'feedback' is to play the feedback audio
        #save the image location to the corresponding command
        if key == 'w':
            minimal_client.send_move_request('move_forward')
            img_loc = '/home/ubuntu/tmp_dir/img_dir/front_new.jpg'
        elif key == 'a':
            minimal_client.send_move_request('turn_left')
            img_loc = '/home/ubuntu/tmp_dir/img_dir/left_new.jpg'
        elif key == 'd':
            minimal_client.send_move_request('turn_right')
            img_loc = '/home/ubuntu/tmp_dir/img_dir/right_new.jpg'
        elif key == 'q':
            minimal_client.send_move_request('move_left')
            img_loc = '/home/ubuntu/tmp_dir/img_dir/left_new.jpg'
        elif key == 'e':
            minimal_client.send_move_request('move_right')
            img_loc = '/home/ubuntu/tmp_dir/img_dir/right_new.jpg'
        elif key == 's':
            minimal_client.send_move_request('move_backward')
            img_loc = '/home/ubuntu/tmp_dir/img_dir/front_new.jpg'
        elif key == 'z':
            minimal_client.send_move_request('look_up')
            img_loc = '/home/ubuntu/tmp_dir/img_dir/left_new.jpg'
        elif key == 'c':
            minimal_client.send_move_request('look_down')
            img_loc = '/home/ubuntu/tmp_dir/img_dir/left_new.jpg'
        elif key == 'x':
            minimal_client.send_move_request('look_straight')
            img_loc = '/home/ubuntu/tmp_dir/img_dir/left_new.jpg'
        elif key =='feedback':
            x, sr = a2n.audio_from_file("/home/ubuntu/transfer_dir/feedback.mp3")
            img_loc = '/home/ubuntu/tmp_dir/img_dir/front_new.jpg'
            sd.play(x, sr)
        
        # Display the image in the pupper and set the old time to the current time
        old_time = time
        disp.show_image(img_loc)

    # Destroy node and shut down
    minimal_client.destroy_node()
    rclpy.shutdown()
   

if __name__ == '__main__':
    main()

