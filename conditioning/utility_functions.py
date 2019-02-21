import os  # used to find files and create paths
import sys # sys.stdout used for inline printing
import cv2 # used for reading images / face detection
from random import choice    # ]
from string import hexdigits # ] used to create random filenames
from collections import namedtuple


classifier_file = "haarcascade_frontalface_default.xml"

if classifier_file not in os.listdir():
    msg = "Unable to find classifier file `{}`.".format(classifier_file)
    raise FileNotFoundError(msg)

# Declare Face datatype used by `face_check` function
Face = namedtuple('Face', ['detected', 'bounding_box', 'target_size'])

def flip_channel_3(image):
    """Reverse the order of the last dimension

    Can be used to convert RGB images to BGR, or vice versa.
    """
    return image[...,::-1]

def face_check(img, bins):
    """Detects the faces in the photo using Haar cascade.

    If more than one face exists, additional faces are discarded.
    Checks if the cropped face image will fit within the bounds of a bin.

    Returns: A namedtuple called Face with attributes:
        detected: bool, 
        bounding_box: List[int], 
        target_size: int
    """
    no_face_found = Face(detected=False, bounding_box=None, target_size=None)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(classifier_file)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Signature: detectMultiScale(image, scaleFactor, minNeighbors)
    # Returns list of rectanges, e.g.
    #     [[x1, y1, h1, w1], 
    #      [x2, y2, h2, w2]]

    if len(faces) != 1:
        return no_face_found
    
    # Extract only the first face bounding box
    face_bb = faces[0]
    x, y, width, height = face_bb

    # Check if the face fits into any bin
    for b in bins:
        target_size, lower_bound, upper_bound = b
        
        good_width  = (width  <= upper_bound and width  > lower_bound)
        good_height = (height <= upper_bound and height > lower_bound)

        if(good_width and good_height):
            return Face(detected=True, bounding_box=face_bb, target_size=target_size)
    
    return no_face_found

# TODO: Combine `scale_adjustment` and `crop_realign` into single function
def scale_adjustment(x, y, crop_width, crop_height, target_width, target_height):
    """Adjusts the crop window to target height and target dimensions"""
    delta_h = target_height - crop_height
    delta_w = target_width - crop_width
    
    delta_y = delta_h // 2
    delta_x = delta_w // 2
    
    y -= delta_y
    x -= delta_x

    if (x < 0): x = 0
    if (y < 0): y = 0
    
    return (x, y, target_width, target_height)

def crop_realign(x, y, crop_width, crop_height, image_width, image_height):
    """Translates crop window to prevent overflow"""
    if (x + crop_width > image_width):
        diff = x + crop_width - image_width
        x -= diff

    if (y + crop_height > image_height):
        diff = y + crop_height - image_height
        y -= diff
    
    if (x < 0): x = 0
    if (y < 0): y = 0
    
    return (x, y, crop_width, crop_height)
        
def generate_random_hex_code(size):
    """Generates a string of random hex digits (a-f, A-F, 0-9)"""
    return "".join([choice(hexdigits) for _ in range(size)])

def crop_image(img, x, y, w, h):
    """Crops image to specified height and width using an anchor (x,y)"""
    return img[y:y+h, x:x+w, :]

def collect_and_save_images(source_path, dest_path, num_images, bins):
    """Looks for faces in images located within given directory.

    If a face is detected, 
        - place it into the proper bin,
        - crop the image to the face bounding box, and
        - save it to the destination
    """
    print("Attempting to collect {n} images...".format(n=num_images))
    
    # build directories
    for b in bins:
        target_size, lower_bound, upper_bound = b
        path = os.path.join(dest_path, "{n}x{n}".format(n=target_size))
        if not os.path.exists(path):
            os.mkdir(path)
    
    good_images = []
    subjects = os.listdir(source_path)

    for sub in sorted(subjects):
        subject_path = os.path.join(source_path, sub)
        image_filenames = os.listdir(subject_path)

        for filename in sorted(image_filenames):
            image_path = os.path.join(subject_path, filename)
            image = cv2.imread(image_path)
            
            face = face_check(image, bins)
            # face_check returns a Face object (namedtuple)

            if(face.detected):
                (x, y, w, h) = face.bounding_box
                ts = face.target_size
                
                image_width = image.shape[1]
                image_height = image.shape[0]
                
                (x, y, w, h) = scale_adjustment(x, y, w, h, ts, ts)
                (x, y, w, h) = crop_realign(x, y, w, h, image_width, image_height)
                cropped = crop_image(image, x, y, w, h)
                good_images.append(face.target_size)
                
                # save image
                image_ext  = ".png"
                image_name = generate_random_hex_code(size=16)
                bin_folder = "{ts}x{ts}".format(ts=face.target_size)
                image_path = os.path.join(dest_path, bin_folder,
                                          image_name + image_ext)

                cv2.imwrite(image_path, cropped)

                # Print out progress in-place
                info_str = "[Current subject: {subj}, "\
                           "Total progress: {n} / {m} ({p:.2f}%)]\r".format(
                                subj=sub, n=len(good_images), m=num_images,
                                p=len(good_images) / float(num_images) * 100)

                sys.stdout.write(info_str)
                sys.stdout.flush()
                
            if(len(good_images) == num_images):
                return good_images

    print("\nUnable to find {n} images. "
          "Returning {m} instead.".format(n=num_images, m=len(good_images)))
    return good_images

def parse_config(config_file):
    # Verify that config_file is in current directory
    if config_file not in os.listdir():
        msg = "Could not find configuration file `{}`".format(config_file)
        raise FileNotFoundError(msg)
    
    # Read lines from the config file; store in list
    with open(config_file, 'r') as f:
        config = f.readlines()
    
    bins = []
    
    for i, line in enumerate(config):
        if (i == 0):
            source_path = line.split(" ")[-1].strip()
            # Note: strip() is used to remove the trailing '\n'
        elif (i == 1):
            destination_path = line.split(" ")[-1].strip()
        elif (i == 2):
            quantity = line.split(" ")[-1].strip()
            quantity = int(quantity)
        elif (i > 3):
            l = line.split(", ")
            l = list(map(int, l))
            bins.append(l)
    
    # Print out parsed configuration:
    print("="*50)
    print("Parsed configuration file: `{}`".format(config_file))
    print("="*50)
    print("Source directory:", source_path)
    print("Destination directory:", destination_path)
    print("Image quantity:", quantity)
    print("Bins:")
    print("          [Target, Minimum, Maximum]")
    print("         ----------------------------")
    for b in bins:
        print("          [{:6d}, {:7d}, {:7d}]".format(*b))
    print("="*50 + "\n")

    return source_path, destination_path, quantity, bins
