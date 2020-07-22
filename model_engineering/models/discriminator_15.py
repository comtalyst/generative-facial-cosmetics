########## Discriminator model ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model
else:
  from tensorflow.keras import layers, Model

###### Constants ######

LATENT_SIZE = 128
IMAGE_SHAPE = (15, 15, 4)

###### Functions ######

def d_block(input_tensor, filters, reduce_times = 2):
  out = layers.Conv2D(filters, 3, padding = 'same')(input_tensor)
  out = layers.LeakyReLU(0.2)(out)
  if reduce_times > 1:
    out = layers.AveragePooling2D(reduce_times)(out)
  return out

def build_model(strategy):
  with strategy.scope():
    # Image input
    image_input = layers.Input(IMAGE_SHAPE, name="input_image")

    # Size: 15x15x4
    x = d_block(image_input, 64)

    # Size: 15x15x64
    x = d_block(x, 128, 3)

    # Size: 5x5x128
    x = layers.Conv2D(256, 3, padding = 'same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)

    # 1-dimensional Neural Network
    class_output = layers.Dense(1, name="output_prob")(x)

    # Make Model
    model = Model(inputs = image_input, outputs = class_output)

    model.summary()

  return model

###### Execution ######

