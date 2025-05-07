import pygame
import os
import random
import math
from shape import Shape

## SHAPE RELATED

# Make functions to generate scale and positions of shapes

# Function that creates a specified number of random shape scales
def random_shape_scale(num_scales, scale_min, scale_max):
    scales = []

    for i in range(num_scales):
        rand_scale = random.randint(math.floor(scale_min), math.floor(scale_max))
        scales.append((rand_scale, rand_scale))

    return scales

# Function that creates sprite group to contain shapes
def create_sprite_group(shape_set, scales, positions):
  # Parameters:
  # shape_set - List of directories to shapes
  # scale - Tuple containing the width and height of the shapes
  # positions - List of tuples containing the x and y coordinates of the center of the shapes

  shapes = pygame.sprite.Group()

  for i in range(len(shape_set)):
      shapes.add(Shape(shape_set[i], positions[i], scales[i]))

  return shapes

## SHAPE SET RELATED

# Function that uses listdir to get paths to all shapes in shapes_folder
def get_shape_images(shapes_folder):
    # Parameters:
    # shapes_folder - directory of folder containing shape images
    shape_images = os.listdir(shapes_folder)
    return shape_images

# Function that gets list of directories to random set of images
def generate_set(shapes_folder, num_shapes):
    # Parameters:
    # shape_images - list with paths to every shape image
    # num_shapes - number of shapes we we want in our set

    shape_images = get_shape_images(shapes_folder)

    shape_set = []

    for i in range(num_shapes):
        rand_index = random.randrange(len(shape_images))
        shape_set.append(f'{shapes_folder}/' + shape_images[rand_index])

    return shape_set

## MODIFYING SHAPES

# Function that rotates shapes
def blitRotate(image, pos, originPos, angle):

  # offset from pivot to center
  image_rect = image.get_rect(topleft=(pos[0] - originPos[0],
                                       pos[1] - originPos[1]))
  offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
  
  # rotated offset from pivot to center
  rotated_offset = offset_center_to_pivot.rotate(-angle)
  
  # rotated image center
  rotated_image_center = (pos[0] - rotated_offset.x,
                          pos[1] - rotated_offset.y)
  
  # get a rotated image
  rotated_image = pygame.transform.rotate(image, angle)
  rotated_image_rect = rotated_image.get_rect(center=rotated_image_center)
  
  return rotated_image, rotated_image_rect


