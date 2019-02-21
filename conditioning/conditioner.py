import os
from utility_functions import collect_and_save_images

# What's the name of our configuration file?
config_file = "conditioner.conf"

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
    print("Source directory:", source_path)
    print("Destination directory:", destination_path)
    print("Image quantity:", quantity)
    print("Bins:")
    print("    [ Target, Minimum, Maximum ]")
    for b in bins:
        print("    ", b)

    return source_path, destination_path, quantity, bins

source_path, destination_path, quantity, bins = parse_config(config_file)

# Check if destination exists:
if not os.path.exists(destination_path):
    os.mkdir(destination_path)

image_dimensions = collect_and_save_images(source_path, destination_path, quantity, bins)

print("\nTotal number of images collected:", len(image_dimensions))
