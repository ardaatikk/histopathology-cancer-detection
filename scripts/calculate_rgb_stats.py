import numpy as np
import argparse
import os

from PIL import Image

parser = argparse.ArgumentParser(description='')
parser.add_argument('--image_folder', default='../images', help='Images directory')

FLAGS = parser.parse_args()

image_folder = FLAGS.image_folder  # Image folder

# Initializing variables rgb_sum and rgb_count to keep track of the sum of RGB values and the total number of pixels processed.
rgb_sum = np.array([0.0, 0.0, 0.0]) 
rgb_count = 0
pixel2_sum = np.array([0.0, 0.0, 0.0]) 

# Loop through all images in the folder
for image_path in os.listdir(image_folder):

    # Open image and convert to RGB
    image = Image.open(os.path.join(image_folder, image_path)) # Opening each image file using PIL's Image.open() method 
    pixels = np.asarray(image, dtype=float)/255. # converting it to a NumPy array for further processing.    
    # print(pixels.shape)
    # print(pixels.dtype)
    pixel2 = pixels**2
    pixel2_sum += np.sum(pixel2, axis=(0, 1))
    # pixels.shape
    # Sum up RGB values
    rgb_sum += np.sum(pixels, axis=(0, 1))  # we defined axis because we want height and width calculated seperately. if we remove it the sum will lead us the wrong calculations to mean and std.
    rgb_count += image.width * image.height # number of the rgb values of an image.
        
# print(rgb_sum)
# print(pixel2_sum)
# Calculate RGB mean and std
rgb_mean = rgb_sum / rgb_count
pixel2_avg = pixel2_sum / rgb_count 
variance = pixel2_avg - rgb_mean**2
rgb_std = np.sqrt(variance)
# print(variance)
# print(rgb_mean)
# print(pixel2_avg)
print('RGB Mean:', rgb_mean)
print('RGB STD:', rgb_std)


