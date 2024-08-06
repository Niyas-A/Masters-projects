########
# Author: Anand Kumar, Ben Klingensmith, Niyas Attasseri
# Name: gui_ssh
#
# Purpose: RobThePupper. GUI for the pupper to move based on the key.txt file sent from the laptop and get feedback.
#          Make sure to have the bringup, gui_service, and gui_client running before launching this.
#
# Usage: This has to launched in the laptop to get feedback and move the pupper.
#        python3 gui_ssh.py <ip_address_of_pupper>
#
# Date: 12 June 2024
########

import tkinter as tk
from tkinter import messagebox, PhotoImage
from PIL import Image, ImageTk
import cv2
from feedback.feedback import get_feedback, color_dict_HSV, camera_to_mask
from time import time

import os
import sys
import numpy as np
import time

count = 0

###
# Name: toggle_fullscreen
# Purpose: This will toggle the fullscreen mode of the GUI.
# Arguments: event (default=None)
###
def toggle_fullscreen(event=None):
    global is_fullscreen
    is_fullscreen = not is_fullscreen  # Toggle state
    root.attributes("-fullscreen", is_fullscreen)

###
# Name: end_fullscreen
# Purpose: This will end the fullscreen mode of the GUI.
# Arguments: event (default=None)
###
def end_fullscreen(event=None):
    global is_fullscreen
    is_fullscreen = False
    root.attributes("-fullscreen", False)
###
# Name: start_puzzle
# Purpose: This will start the puzzle by loading the mask image and the current image.
# Arguments: None
# Global Variables: mask_image, hidden_quadrants, curr_image
###
def start_puzzle():
    global mask_image, hidden_quadrants, curr_image

    #update the camera image shown in the GUI
    update_puzzle_image()

    # set the hidden quadrants to all 4
    hidden_quadrants = [0, 1, 2, 3]
    if not level.get():
        messagebox.showinfo("Select Level", "Please select a difficulty level!")
        return
    
    # Load the mask image and apply the mask only for green color
    image_path = f"mask_true.png"
    try:
        # read the image
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            raise IOError("Image file not found")
        mask_image = cv_img.copy()
        color_name = "green"
        color = color_dict_HSV[color_name]
        # apply the mask for the green color
        mask_image = camera_to_mask(mask_image, color_lower = color[1], color_upper = color[0])
    except IOError as e:
        messagebox.showerror("Image Load Error", str(e))
        return
    
    #update the image shown in the GUI
    update_image()

    mask_level_label.config(text=f"Mask Level: {level.get()}")
####
# Name: generate_feedback
# Purpose: This will generate the feedback based on the current image and the mask image.
# Arguments: None
# Global Variables: curr_image, mask_image, count, hidden_quadrants
###
def generate_feedback():
    global curr_image, mask_image, count, hidden_quadrants

    # Check if the level is selected and the puzzle is started
    if not level.get():
        messagebox.showinfo("Select Level and start puzzle", "Please select a difficulty level and start puzzle!")
        return
    
    count += 1  # Increment count each time feedback is generated

    #update the camera image variable
    update_puzzle_image()

    # Get the feedback based on the current image and the mask image
    color_name = "green"
    color = color_dict_HSV[color_name]
    img_feedback, hidden_quadrants = get_feedback(curr_image, mask_image, hidden_quads=hidden_quadrants, color_lower=color[1], color_upper=color[0])

    # update the image shown in the GUI
    update_image()
    feedback_message = f"Check: {count} - Feedback: {img_feedback}"

    # If the feedback is the solution, generate the feedback and save it to the key.txt file
    time_now = time.time()
    #save it to file key.txt
    with open('key.txt', 'w') as f:
        f.write(str(time_now) + '\n')
        f.write('f')

    #rsync to pupper
    os.system('rsync key.txt ubuntu@' + ip + ':/home/ubuntu/transfer_dir/key.txt')

    # Update the feedback text box in the GUI
    update_feedback(feedback_message)

####
# Name: update_puzzle_image
# Purpose: This will update the camera image in the GUI.
# Arguments: None
# Global Variables: image_label, curr_image
###
def update_puzzle_image():
    global image_label, curr_image

    #update the camera image variable by copying the image from the pupper and saving it as camera_frame.png
    #try-except block to handle the case when the image is not found
    try:
        os.system('rsync ubuntu@' + ip + ':/home/ubuntu/transfer_dir/camera_image.png camera_frame.png')
        time.sleep(0.2)
        image_path = "camera_frame.png"
        cv_img = cv2.imread(image_path)
        curr_image = cv_img.copy()
        if cv_img is None:
            raise IOError("Image file not found")

    except:
        time.sleep(0.2)
        os.system('rsync ubuntu@' + ip + ':/home/ubuntu/transfer_dir/camera_image.png camera_frame.png')
        image_path = "camera_frame.png"
        cv_img = cv2.imread(image_path)
        curr_image = cv_img.copy()
        if cv_img is None:
            raise IOError("Image file not found")

####
# Name: update_image
# Purpose: This will update the image in the GUI by masking the quadrants based on the hidden quadrants.
# Arguments: None
# Global Variables: curr_image, image_label, mask_image, hidden_quadrants
###

def update_image():
    global curr_image, image_label, mask_image, hidden_quadrants

    # Mask the quadrants based on the hidden quadrants
    quad_flags = [True, True, True, True]
    for quad in hidden_quadrants:
        quad_flags[quad] = False

    # Mask the quadrants in the image
    masked_image = mask_quadrants(curr_image, mask_image, quad_flags=quad_flags, alpha=0.2)

    # Resize the image and update the image in the GUI
    resized_cv_img = cv2.resize(masked_image,( (1280//2, 720//2)))  # Resize the image
    cv_img_rgb = cv2.cvtColor(resized_cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img_rgb)
    img = ImageTk.PhotoImage(pil_img)
    image_label.configure(image=img)
    image_label.image = img


####
# Name: mask_quadrants
# Purpose: This will mask the quadrants in the image based on the hidden quadrants.
# Arguments: live_image, mask_image, quad_flags, alpha
# Returns: result
###

def mask_quadrants(live_image, mask_image, quad_flags = [True, True, True, True], alpha = 0.5):
    assert live_image.shape[0] == mask_image.shape[0] and live_image.shape[1] == mask_image.shape[1]
    height, width = mask_image.shape

    # Make a copy of the live image and convert the mask image to float32
    result = np.float32(np.copy(live_image))
    mask_image = np.float32(mask_image)/255

    #scale from 0-1 to alpha-1
    mask_image = (1 - alpha*2) * mask_image + alpha*2
    #convert to 3 channels
    mask_image = np.repeat(mask_image[:, :, np.newaxis], 3, axis=2)
    #create a mask of zeros
    zeros_image = np.ones_like(mask_image)*alpha

    # Mask the quadrants based on the hidden quadrants
    for i in range(2):
        for j in range(2):
            if quad_flags[i * 2 + j]:
                result[i * height // 2: (i + 1) * height // 2, j * width // 2: (j + 1) * width // 2] *= mask_image[i * height // 2: (i + 1) * height // 2, j * width // 2: (j + 1) * width // 2]
            else:
                result[i * height // 2: (i + 1) * height // 2, j * width // 2: (j + 1) * width // 2] *= zeros_image[i * height // 2: (i + 1) * height // 2, j * width // 2: (j + 1) * width // 2]

    # Convert the result to uint8
    return np.uint8(result)

####
# Name: update_feedback
# Purpose: This will update the feedback text box in the GUI with the new message.
# Arguments: message
# Global Variables: feedback_text
###

def update_feedback(message):
    feedback_text.config(state=tk.NORMAL)
    feedback_text.delete('1.0', tk.END)
    feedback_text.insert(tk.END, message)
    feedback_text.config(state=tk.DISABLED)

####
# Name: set_level
# Purpose: This will set the difficulty level in the GUI.
# Arguments: selected_level
# Global Variables: level, easy_button, medium_button, hard_button
###

def set_level(selected_level):
    level.set(selected_level)
    for button, lvl in zip([easy_button, medium_button, hard_button], ["Easy", "Medium", "Hard"]):
        button.config(relief=tk.SUNKEN if level.get() == lvl else tk.RAISED)

####
# Name: quit_application
# Purpose: This will quit the application.
# Arguments: None
###
def quit_application():
    root.destroy()

####
# Name: key_press
# Purpose: This will handle the key press event and save the key press to the key.txt file.
# Arguments: event
# Global Variables: ip
###

def key_press(event):
    # Get the key pressed
    key = event.keysym

    # If the key corresponds to a move command, save it to the key.txt file and rsync it to the pupper
    if key in {'w', 'a', 's', 'd', 'z', 'c', 'x', 'f', 'q', 'e'}:
        print(f"Move: {key}")
        time_now = time.time()
        #save it to file key.txt
        with open('key.txt', 'w') as f:
            f.write(str(time_now) + '\n')
            f.write(key)

        #rsync to pupper
        os.system('rsync key.txt ubuntu@' + ip + ':/home/ubuntu/transfer_dir/key.txt')


if __name__ == "__main__":

    global image_label
    ip = sys.argv[1]

    # Main window for the tkinter GUI
    root = tk.Tk()
    root.title("Rob the Builder")
    root.geometry('1000x600')

    # Fullscreen mode
    is_fullscreen = True
    root.attributes("-fullscreen", is_fullscreen)
    root.bind("<F11>", toggle_fullscreen)
    root.bind("<Escape>", end_fullscreen)

    # Heading
    heading_label = tk.Label(root, text="Rob the Builder", font=("Arial", 24))
    heading_label.pack(pady=20)

    # Difficulty level options
    level = tk.StringVar()  # To store the selected difficulty level
    frame = tk.Frame(root)
    frame.pack(pady=20)

    #track key press
    root.bind("<Key>", key_press)
    
    # Difficulty level buttons
    easy_button = tk.Button(frame, text="Easy", command=lambda: set_level("Easy"))
    easy_button.pack(side=tk.LEFT)

    medium_button = tk.Button(frame, text="Medium", command=lambda: set_level("Medium"))
    medium_button.pack(side=tk.LEFT)

    hard_button = tk.Button(frame, text="Hard", command=lambda: set_level("Hard"))
    hard_button.pack(side=tk.LEFT)

    # Start button
    start_button = tk.Button(root, text="Start the Puzzle", command=start_puzzle)
    start_button.pack(pady=20)

    # Quit button
    quit_button = tk.Button(root, text="Quit", command=quit_application)
    quit_button.pack(side=tk.RIGHT, pady=20, padx=10)

    # Generate feedback button
    feedback_button = tk.Button(root, text="Generate Feedback", command=generate_feedback)
    feedback_button.pack(pady=10)
    
    # Mask level label
    mask_level_label = tk.Label(root, text="Mask Level: Not Selected", font=("Arial", 16))
    mask_level_label.pack(pady=5)

    # Container for the image and feedback text
    container = tk.Frame(root)
    container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

    # Image label
    image_label = tk.Label(container, height=1280, width=720)
    image_label.pack(side=tk.LEFT, padx=10)

    # Image heading
    image_heading = tk.Label(container, text="Mask Image", font=("Arial",14))
    image_heading.pack(side=tk.TOP, padx=10)

    # Feedback text box
    feedback_text = tk.Text(container, height=20, width=50, font=("Arial", 16))
    feedback_text.pack(side=tk.RIGHT, padx=10)
    feedback_text.config(state=tk.DISABLED)

    # Update the camera image in the GUI
    try:
        os.system('rsync ubuntu@' + ip + ':/home/ubuntu/transfer_dir/camera_image.png camera_frame.png')
        image_path = "camera_frame.png"
        image = Image.open(image_path)
        img = ImageTk.PhotoImage(image)
    except:
        os.system('rsync ubuntu@' + ip + ':/home/ubuntu/transfer_dir/camera_image.png camera_frame.png')
        image_path = "camera_frame.png"
        image = Image.open(image_path)
        img = ImageTk.PhotoImage(image)

    # Start the GUI
    root.mainloop()
