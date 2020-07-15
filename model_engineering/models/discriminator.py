########## Discriminator model ##########

###### Imports ######

import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.api._v2.keras import layers, Model

###### Constants ######

LATENT_SIZE = 256
IMAGE_SHAPE = (360, 360, 4)

###### Functions ######

def d_block(input_tensor, filters, reduce_times = 2):
  out = layers.Conv2D(filters, 3, padding = 'same')(input_tensor)
  out = layers.LeakyReLU(0.2)(out)
  if reduce_times > 1:
    out = layers.AveragePooling2D(reduce_times)(out)
  return out

def build_model():
  # Image input
  image_input = layers.Input(IMAGE_SHAPE)

  # Size: 360x360x4
  x = d_block(image_input, 8)

  # Size: 180x180x8
  x = d_block(x, 16)

  # Size: 90x90x16
  x = d_block(x, 32)

  # Size: 45x45x32
  x = d_block(x, 64, 3)

  # Size: 15x15x64
  x = d_block(x, 128, 1)

  # Size: 15x15x128
  x = d_block(x, 256, 3)

  # Size: 5x5x256
  x = layers.Conv2D(512, 3, padding = 'same')(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Flatten()(x)

  # 1-dimensional Neural Network
  class_output = layers.Dense(1)(x)

  # Make Model
  model = Model(inputs = image_input, outputs = class_output)

  model.summary()

  return model

###### Execution ######

