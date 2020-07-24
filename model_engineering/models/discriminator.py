########## Discriminator model ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
import os
import h5py
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model, models
else:
  from tensorflow.keras import layers, Model

###### Class Content ######
class Discriminator:

  ###### Constants ######

  LATENT_SIZE = 256
  FINAL_IMAGE_SHAPE = (360, 360, 4)
  MAX_PROGRESS = 5
  model = None

  current_progress = 0              # for GAN progressive training

  ###### Constructor ######

  def __init__(self, strategy):
    self.model = self.build_model(strategy)

  ###### Public Methods ######

  def progress(self, strategy):
    self.current_progress += 1
    self.model = self.progress_model(self.model, strategy)
  
  def save(self, epoch, dir = None):
    model = self.model

    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    fname = "discriminator" + "-p_" + str(self.current_progress) + "-e_" + str(epoch) + ".h5"
    model.save(os.path.join(dir, fname))
  
  # either specify dir/fname or path (path takes priority)
  def load(self, its_progress, dir = None, fname = None, path = None):
    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    if fname == None and path == None:
      raise ValueError("Please specify either fname or path")
    if path == None:
      path = os.path.join(dir, fname)
    self.current_progress = its_progress
    self.model = models.load_model(h5py.File(path, 'r'))
    self.model.summary()

  ###### Functions ######

  def d_block(self, input_tensor, filters, reduce_times = 2):
    out = layers.Conv2D(filters, 3, padding = 'same')(input_tensor)
    out = layers.LeakyReLU(0.2)(out)
    if reduce_times > 1:
      out = layers.AveragePooling2D(reduce_times)(out)
    return out
  
  def input_block(self, shape, filters):
    self.INPUT_BLOCK_LEN = 3
    # Image input
    image_input = layers.Input(shape)
    # expand filters without affecting other variables (this layer will be deleted in the next generation)
    x = layers.Conv2D(filters, 1, padding = 'same')(image_input)
    x = layers.LeakyReLU(0.2)(x)
    return x, image_input
    
  def build_model(self, strategy):
    d_block = self.d_block
    input_block = self.input_block

    with strategy.scope():
      # Image input
      self.image_shape = (5, 5, 4)
      x, image_input = input_block((5, 5, 4), 256)
      # Size: 5x5x256

      # convert to prob for decision using 1-dimensional Neural Network
      x = d_block(x, 512, reduce_times = 1)
      x = layers.Flatten()(x)
      x = layers.Dense(1)(x)

      # Make Model
      model = Model(inputs = image_input, outputs = x)

      model.summary()

    return model
  
  ## GAN progressive training: add layers
  def progress_model(self, model, strategy):
    d_block = self.d_block
    input_block = self.input_block
    current_progress = self.current_progress

    if current_progress >= self.MAX_PROGRESS:
      print("Maximum progress reached!")
      return model
    current_progress += 1

    with strategy.scope():
      # cut the input block from the previous generation
      # the preferred input shape for this should be the last "after-input" shape
      y = model.layers[self.INPUT_BLOCK_LEN].output

      if current_progress == 1:
        self.image_shape = (15, 15, 4)
        x, image_input = input_block((15, 15, 4), 64)
        # Size: 15x15x64
        x = d_block(x, 128, 1)
        # Size: 15x15x128
        x = d_block(x, 256, 3)
        # Size: 5x5x256
        # should now be match with y's preferred input shape

      elif current_progress == 2:
        self.image_shape = (45, 45, 4)
        x, image_input = input_block((45, 45, 4), 32)
        # Size: 45x45x32
        x = d_block(x, 64, 3)
        # Size: 15x15x64

      elif current_progress == 3:
        self.image_shape = (90, 90, 4)
        x, image_input = input_block((90, 90, 4), 16)
        # Size: 90x90x16
        x = d_block(x, 32, 2)
        # Size: 45x45x32

      elif current_progress == 4:
        self.image_shape = (180, 180, 4)
        x, image_input = input_block((180, 180, 4), 8)
        # Size: 180x180x8
        x = d_block(x, 16, 2)
        # Size: 90x90x16

      elif current_progress == 5:
        self.image_shape = (360, 360, 4)
        x, image_input = input_block((360, 360, 4), 4)
        # Size: 360x360x4
        x = d_block(image_input, 8)
        # Size: 180x180x8
      
      # continue on with the model
      for i in range(self.INPUT_BLOCK_LEN, len(model.layers)):
        x = model.layers[i](x)

      # Make new model on top of the old model
      model = Model(inputs = image_input, outputs = x)

      model.summary()
    
    # return progressed model
    print("The model progressed to level " + str(current_progress))
    return model


