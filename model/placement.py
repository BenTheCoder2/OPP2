# U net image segmentation
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


pixels = []
w,h,c = 1000, 1000, 3 #placeholder numbers
num_filters=  128 # default neuron layers (2^7-n) for mid level feature detection


ckpt_dir = r'C:\Users\Benson\OPP\ML\checkpoint\placement'



def encoderBlock(num_filters, input): # 2-3 convolutional layers
  x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='valid', activation='relu')(input)
  x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='valid', activation='relu')(x)

  x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
  return x

def bottleneck(num_filters, input):
  x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='valid', activation='relu') (input)
  x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='valid', activation='relu') (x)
  return x

def decoderBlock(num_filters, skip, input): # 2-3 convolutional layers
  x = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=2, strides=1, padding='valid')(input) # default stride
  c = tf.image.resize(skip, size=(x.shape[1], x.shape[2]))
  x = tf.keras.layers.Concatenate()([x, c])

  x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
  x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
  return x


# 3 encode blocks
def UNet(input_size, num_classes):

  input = tf.keras.layers.Input(input_size)


  x1 = encoderBlock(32, input)
  x2 = encoderBlock(64, x1)
  x3 = encoderBlock(128, x2)

  x4 = bottleneck(256, x3)

  x5 = decoderBlock(128, x3, x4)
  x6 = decoderBlock(64, x2, x5)
  x7 = decoderBlock(32, x1, x6)

  x8 = tf.keras.layers.GlobalAveragePooling2D()(x7)
  output = tf.keras.layers.Dense(num_classes, activation='softmax')(x8)

  model = tf.keras.models.Model(inputs = input,  
                                outputs = output,  
                                name = 'U-Net') 
  
  return model



'''
if __name__ == '__main__':
  model = UNet(input_size = (1000,1000,3), num_classes = 37)
  model.summary()
'''

# convolutional encoder for sequence and decoder for placement planning

def predict(model, x):
  prediction =  model.predict(x)
  return prediction



def save_weights_UNET(model):
  model.save_weights(ckpt_dir)

def load_weights_UNET(model):
    model.load_weights(ckpt_dir)
  