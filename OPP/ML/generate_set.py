import os
import random


# Use listdir to get paths to all shapes in shapes_folder
def get_shape_images(shapes_folder):
    shape_images = os.listdir(shapes_folder)
    return shape_images


# Function for getting random set of images
# How images are implemented is based on whether its in gym or pygame
def generate_set(shape_images, num_shapes):
    # Parameters:
    # shape_images - list with paths to every shape image
    # num_shapes - number of shapes we want a set of

    shape_set = []

    for i in range(num_shapes):
       # see if getting two of the same images in a single iteration will affect the model’s learning
        rand_index = random.randrange(len(shape_images))
        shape_set.append('shapes/' + shape_images[rand_index])

    return shape_set