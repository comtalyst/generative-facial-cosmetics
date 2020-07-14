########## Discriminator model ##########

###### Imports ######

import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.api._v2.keras import layers, Model

###### Constants ######

LATENT_SIZE = 64
IMAGE_SHAPE = (370, 370, 4)

###### Functions ######

def d_block(input_tensor, filters):
  out = layers.Conv2D(filters, IMAGE_SHAPE[2], padding = 'same')(input_tensor)
  out = layers.LeakyReLU(0.2)(out)
  out = layers.AveragePooling2D()(out)
  return out

def build_model():          # COPIED DUMMY MODEL, NEED READJUSTMENT FOR OUR INPUT, OUTPUT
  # Image input
  image_input = layers.Input(IMAGE_SHAPE)

  # Size: 64x64x3
  x = d_block(image_input, 8)

  # Size: 32x32x8
  x = d_block(x, 16)

  # Size: 16x16x16
  x = d_block(x, 32)

  # Size: 8x8x32
  x = d_block(x, 64)

  # Size: 4x4x64
  x = layers.Conv2D(128, IMAGE_SHAPE[2], padding = 'same')(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Flatten()(x)

  # 1-dimensional Neural Network
  class_output = layers.Dense(1)(x)

  # Make Model
  model = Model(inputs = image_input, outputs = class_output)

  model.summary()

  return model

###### Execution ######

