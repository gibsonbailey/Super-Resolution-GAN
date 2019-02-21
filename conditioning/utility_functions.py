import numpy as np
import os
import cv2
import math
import matplotlib.pyplot as plt
import string
import random


# Reverses the order of channel 3.
# RGB to BGR or BGR to RGB
def flip_channel_3(image):
    return image[...,::-1]


# Loads a number of images from the vggface2 dataset
def load_images(image_quantity):
    path = '../data/vggface2/test/' # Shouldn't we use the path we loaded from the config?

    print("Loading %d images..." % image_quantity)
    
    subjects = os.listdir(path)

    images = []
    
    for i, sub in enumerate(subjects):
        for j, image in enumerate(os.listdir(path + sub)):
            im = cv2.imread(path + sub + "/" + image)
            im = im[...,::-1]
            images.append(im)
            
            if(len(images) % max(1, int(image_quantity / 20)) == 0):
                print("%f%%" % (len(images) / image_quantity))
            
            if(len(images) >= image_quantity):
                images = np.array(images)
                return images


# Detects the faces in the photo using Haar cascade.
# If more than one face exists, it is discarded.
# Checks if the cropped face image will fit within the bounds of a bin.
# Returns success_value, bounding_box_coordinates, bin_number
def face_check(img, bins):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) != 1:
        return (0, None, None)
    faces = faces[0]

    for b in bins:
        upper_bound = b[2]
        lower_bound = b[1]
        
        # faces = x, y, w, h
        if(faces[2] <= upper_bound and faces[2] > lower_bound and faces[3] <= upper_bound and faces[3] > lower_bound):
            return (1, faces, b[0])
    
    return (0, None, None)


# Adjusts the scale of the bounding box to desired height and width dimensions
def scale_adjustment(x, y, w_c, h_c, w_d, h_d):
    delta_h = h_d - h_c
    delta_w = w_d - w_c
    
    delta_y = -delta_h // 2
    delta_x = -delta_w // 2
    
    y += delta_y
    x += delta_x
    
    return (x, y, w_d, h_d)


# Translates crop bounding box to eliminate overflow
def crop_realign(x, y, w_c, h_c, w_i, h_i):
    if(x + w_c > w_i):
        diff = x + w_c - w_i
        x -= diff
        
    if(y + h_c > h_i):
        diff = y + h_c - h_i
        y -= diff
    
    if(x < 0):
        x = 0
    
    if(y < 0):
        y = 0
    
    return (x, y, w_c, h_c)
        

# Generates a string of random hex digits (a-f, A-F, 0-9)
def generate_random_hex_code(length):
    s = string.hexdigits
    output = ""
    
    for i in range(length):
        output += s[random.randint(0, len(s) - 1)]

    return output


# Crops image given bounding box
def crop_image(img, x, y, w, h):
    return img[y:y+h,x:x+w,:]


# Legacy bins creation function (NEEDS UPDATING)
def create_bins(lower, upper, interval):
    bins = []
    while lower <= upper:
        bins.append(lower)
        lower += interval
    return bins


# Inspects images at a given source path.
# Looks for faces within each image.
# Sorts the images into the proper bin if face is detected.
# Crops image to face bounding box.
# Saves image to destination path.
def collect_and_save_images(source_path, dest_path, image_quantity, bins):

    print("Inspecting %d images..." % image_quantity)
    
    subjects = os.listdir(source_path)
    
    prog_count = 0
    
    good_images = []
    
    # build directories
    for b in bins:
        path = dest_path + "%dx%d" % (b[0], b[0])
        if not os.path.exists(path):
            os.mkdir(path)
    
    for i, sub in enumerate(subjects):
        for j, filename in enumerate(os.listdir(source_path + sub)):
            image = cv2.imread(source_path + sub + "/" + filename)
            
            prog_count += 1
            
            if(prog_count % max(1, int(image_quantity / 20)) == 0):
                print("%f%%" % (prog_count / image_quantity * 100))
            
            check = face_check(image, bins)
            detected = check[0]
            if(detected):
                (x, y, w, h) = check[1]
                desired_dimension = check[2]
                
                image_width = image.shape[1]
                image_height = image.shape[0]
                
                (x, y, w, h) = scale_adjustment(x, y, w, h, desired_dimension, desired_dimension)
                (x, y, w, h) = crop_realign(x, y, w, h, image_width, image_height)
                cropped = crop_image(image, x, y, w, h)
                good_images.append(desired_dimension)
                if prog_count % int(image_quantity / 20) == 0:
                    print("Image count:", len(good_images))
                
                # save image
                p = dest_path + "%dx%d/" % (desired_dimension, desired_dimension)
                cv2.imwrite(p + generate_random_hex_code(12) + ".jpg", cropped)
                
            if(prog_count >= image_quantity):
                return good_images

