import numpy as np
import os
import cv2
import math
import string
import random
from utility_functions import collect_and_save_images

print("\n\n\n\n\n\n\n\n\n")
config_file = "conditioner_config"


config_file = open(config_file, "r")
config = config_file.readlines()

bins = []

for i, line in enumerate(config):
    if(i == 0):
        source_path = line.split(" ")[2][0:-1]
    elif(i == 1):
        destination_path = line.split(" ")[2][0:-1]
    elif(i == 2):
        quantity = int(line.split(" ")[2][0:-1])
    elif(i > 3):
        l = line.split(", ")
        for a in range(3):
            l[a] = int(l[a])          
        bins.append(l)
config_file.close()


print("Source directory:", source_path)
print()
print("Destination directory:", destination_path)
print()
print("Image quantity:", quantity)
print()
print("Bins:")
print()
print("[ Target, Minimum, Maximum ]")
for b in bins:
    print(b)

if not os.path.exists(destination_path):
    os.mkdir(destination_path)

print("\n\n\n\n\n\n")
image_dimensions = collect_and_save_images(source_path, destination_path, quantity, bins)

print("Total number of images collected:", len(image_dimensions))
