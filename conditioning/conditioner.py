import os
from utility_functions import parse_config, collect_and_save_images

# What's the name of our configuration file?
config_file = "conditioner.conf"
source_path, destination_path, quantity, bins = parse_config(config_file)

# Check if destination exists:
if not os.path.exists(destination_path):
    os.mkdir(destination_path)

image_sizes = collect_and_save_images(
    source_path, destination_path,
    quantity, bins)

print("\nTotal number of images collected:", len(image_sizes))
