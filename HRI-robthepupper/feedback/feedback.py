########
# Author: Anand Kumar, Ben Klingensmith, Niyas Attasseri
# Name: feedback.py
#
# Purpose: RobThePupper. Support functions for the feedback system.
#
# Usage: This script is used to compare the camera image to a mask and provide feedback to the user.
#        python feedback.py
#        (This script is not meant to be run directly, but rather imported into another script.)
# Date: 12 June 2024
########

import cv2
import numpy as np

# When calculating the percent of pixels matching between two images, 
# use these thresholds and labels to describe how close they are
similarity_thresholds = np.array([.8, .9, .95])-0.35
similarity_threshold_labels = ["not similar to", "similar to", "very similar to", "exactly like"]

# Quandrants are ordered left to right then top to bottom
quadrant_names = ["top left", "top right", "bottom left", "bottom right"]

# Percent of pixels needed to match mask quadrant or full mask to count as a successful solution
success_threshold = .95

# HSV color dictionary taken from https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv
color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}

# Convert similarity percent into string descriptor based on thresholds
def threshold_perc(perc_similar):
    thresholds_passed = np.sum(perc_similar > similarity_thresholds)
    similarity_description = similarity_threshold_labels[thresholds_passed]
    return similarity_description

# Calculate percent shared pixels for each quadrant pair between 2 images
def compare_quads(im1, im2):
    # Expects images be square and same size
    side_len = im1.shape[0]
    half_len = side_len // 2
    assert side_len == im2.shape[0], "Images must be the same size"
    comp_results = []
    for row in [0,half_len]:
        for col in [0,half_len]:
            new_mask_quad = im1[row:row + half_len, col:col + half_len]
            correct_mask_quad = im2[row:row + half_len, col:col + half_len]
            if np.mean(correct_mask_quad) < .02:
                comp_results.append(1)
            else:
                comp_results.append(np.mean(new_mask_quad[correct_mask_quad != 0]))
    return comp_results


# Move a 2d image 
# diff: int tuple pair on number of pixels to move horizontally and vertically.
def move_mask(im, diff):
    pad = max(abs(diff[0]), abs(diff[1]))
    # Pads with 2's which aren't in the mask, so padding will never match with any part of the mask
    padded_im = np.pad(im, pad, constant_values = 2)
    
    (im_x, im_y) = im.shape[0], im.shape[1]
    return padded_im[pad - diff[0]: pad + im_x - diff[0], pad - diff[1]: pad + im_y - diff[1]]

# Translate move (numpy array 2 int elements) into a readable text explanation
def explain_move(move, im_shape):
    x_move = move[1]
    y_move = move[0]
    
    x_dir = "right" if x_move > 0 else "left"
    y_dir = "down" if y_move > 0 else "up"

    # Calculate distance to move as percent of image dimensions
    x_perc = np.rint(100 * abs(x_move / im_shape[1])).astype(int)
    y_perc = np.rint(100 * abs(y_move / im_shape[0])).astype(int)

    return f"{x_perc}% {x_dir} and {y_perc}% {y_dir}"
    

# Initiate ORB detector
orb = cv2.ORB_create()

####
# Function: check_close_quad
#
# Purpose: Comparing two images, look for keypoint pairs between them
# Test if moving the first image to match keypoint pairs would make any of the 
# hidden image quadrants match. Only checks for translations, not rotations
# n is then number of keypoint pair translations to test
# NOTE: matching performs badly with ~<200x200 images
#
# Arguments: im1, im2: 2D numpy arrays representing images
# Returns: string describing the best move to make the images more similar
####
def check_close_quad(im1, im2, hidden_quads = [0,1,2,3], n = 5):
    # find the keypoints and descriptors with ORB
    # Method taken from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    kp1, des1 = orb.detectAndCompute(im1,None)
    kp2, des2 = orb.detectAndCompute(im2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
     
    # Match descriptors.
    matches = bf.match(des1,des2)
     
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    top_score = 0
    top_quad = 0
    top_move = (0,0)
    for i in range(min(n, len(matches))):
        # Calculate keypoint distance between images for one match
        kp1Idx = matches[i].queryIdx
        kp2Idx = matches[i].trainIdx
        move = np.flip(np.rint(kp2[kp2Idx].pt).astype(int) - np.rint(kp1[kp1Idx].pt).astype(int))
        quad_scores = compare_quads(move_mask(im1, move), im2)
        # Calc best quad match
        for quad in hidden_quads:
            if quad_scores[quad] > top_score:
                top_score = quad_scores[quad]
                top_quad = quad
                top_move = move
    threshold_text = threshold_perc(top_score)
    if threshold_text == "not similar to":
        return f"I didn't detect any nearby matches"
    move_text = explain_move(top_move, im1.shape)
    return f"If you move the image {move_text}, I think the {quadrant_names[top_quad]} quadrant would look {threshold_text} the mask."

####
# Function: camera_to_mask
#
# Purpose: Convert an RGB image to a binary mask based on a specified color range
# Range of color for mask creation defaulting to red from lab1
#
# Arguments: rgb_im: 3D numpy array representing an RGB image
#            color_lower: 3 element list of lower bounds for HSV color range
#            color_upper: 3 element list of upper bounds for HSV color range
# Returns: 2D numpy array representing a binary mask of the specified color
####
def camera_to_mask(rgb_im, color_lower = [164, 60, 100], color_upper = [179, 205, 255]):
	# Convert the current frame from RGB to HSV
    hsv = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2HSV)

    # Create numpy arrays from the boundaries
    color_lower = np.array(color_lower,dtype=np.uint8)
    color_upper = np.array(color_upper,dtype=np.uint8)   
        
    color_mask = cv2.inRange(hsv,color_lower,color_upper)
    blurred = cv2.blur(color_mask, (5,5))
    _, bw = cv2.threshold(blurred,127,255,cv2.THRESH_OTSU)

    # #Invert to have mask capture pixels that aren't of the detected color
    # inverted = np.ones(bw.shape) * 255
    # inverted[bw == 255] = 0
    return bw

####
# Function: get_feedback
#
# Purpose: Compare the RGB image from the robot's camera to a specified mask
# Return a message based on the comparison and which quadrants have been revealed
# Get feedback from robot by comparing the rgb image from its camera to the specified mask
# hidden quads stores which quadrants haven't been revealed yet.
# Can optionally accept color_lower and color_upper to be passed to camera_to_mask
#
# Arguments: rgb_im: 3D numpy array representing an RGB image
#            mask: 2D numpy array representing a binary mask
#            hidden_quads: list of integers representing which quadrants haven't been revealed
#            color_lower: 3 element list of lower bounds for HSV color range
#            color_upper: 3 element list of upper bounds for HSV color range
# Returns: string message describing the comparison and list of hidden quadrants
###

def get_feedback(rgb_im, mask, hidden_quads = [0,1,2,3], **kwargs):
    camera_mask = camera_to_mask(rgb_im, **kwargs)
    scores = np.array(compare_quads(camera_mask, mask))
    successes = np.mean(scores) >= success_threshold
    if successes:
        return "You found the solution, congrats!", []

    if not hidden_quads:
        return "You've uncovered all the quadrants, I can't help you any further.", []
    for quad in hidden_quads:
        if scores[quad] >= success_threshold:
            hidden_quads.remove(quad)
            return f"You uncovered the {quadrant_names[quad]} quadrant.", hidden_quads
    return check_close_quad(camera_mask, mask, hidden_quads), hidden_quads

####
# Function: save_new_mask
#
# Purpose: Takes RGB image, converts it to a mask, and saves at the specified location
# Can optionally accept color_lower and color_upper to be passed to camera_to_mask
#
# Arguments: rgb_im: 3D numpy array representing an RGB image
#            save_path: string representing the file path to save the mask
#            color_lower: 3 element list of lower bounds for HSV color range
#            color_upper: 3 element list of upper bounds for HSV color range
# Returns: None
####
def save_new_mask(rgb_im, save_path, **kwargs):
    camera_mask = camera_to_mask(rgb_im, **kwargs)
    cv2.imwrite(save_path, camera_mask)

if __name__ == '__main__':

    #This laods one of my example images and saves a mask of the green in it
    # This only needs to be run once to create a new puzzle mask
    im1 = cv2.imread("camera_frame.png")
    color_name = "green"
    color = color_dict_HSV[color_name]
    save_new_mask(im1, f"1_mask_{color_name}.png", color_lower = color[1], color_upper = color[0])

    # This then loads another image to compare and compares it to a specified mask. 
    # This would be run everytime the user gets the robot's feedback. Make sure that the color the mask was 
    # created with is the same color used here.
    # Also the hidden_quads needs to be updated as the player uncovers the quadrants (i.e. it's [] once they've found them all)
    im2 = cv2.imread("camera_frame.png")
    mask = cv2.imread(f"1_mask_{color_name}.png", cv2.IMREAD_UNCHANGED)
    print(get_feedback(im2, mask, hidden_quads = [0, 1, 2, 3], color_lower = color[1], color_upper = color[0]))

    # You may want to edit get_feedback to change the state of the game depending on the results, 
    # ex: when the player wins reset the game, or you need to store which quadrants are still hidden 
    # and when the player uncovers a quadrant update that information