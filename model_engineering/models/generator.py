########## Generator model ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
import os
import h5py
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model, models
else:
  from tensorflow.keras import layers, Model, models

###### Class Content ######
class Generator:

  ###### Constants ######

  LATENT_SIZE = 256
  FINAL_IMAGE_SHAPE = (360, 360, 4)
  MAX_PROGRESS = 5
  model = None
  image_shapes = {
    0: (5, 5, 4),
    1: (15, 15, 4),
    2: (45, 45, 4),
    3: (90, 90, 4),
    4: (180, 180, 4),
    5: (360, 360, 4)
  }

  current_progress = 0              # for GAN progressive training

  ###### Constructor ######

  def __init__(self, strategy):
    self.model = self.build_model(strategy)
  
  ###### Public Methods ######

  def progress(self, strategy):
    self.model = self.progress_model(self.model, strategy)
    self.current_progress += 1
  
  # save to SavedModel
  def save(self, epoch, strategy, dir = None):
    model = self.model

    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    fname = "generator" + "-p_" + str(self.current_progress) + "-e_" + str(epoch)
    with strategy.scope():
      model.save(os.path.join(dir, fname))

  # either specify dir/fname or path (path takes priority)
  def load(self, its_progress, strategy, dir = None, fname = None, path = None):
    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    if fname == None and path == None:
      raise ValueError("Please specify either fname or path")
    if path == None:
      path = os.path.join(dir, fname)
    self.current_progress = its_progress
    with strategy.scope():
      self.model = models.load_model(path)
      self.model.summary()
    self.image_shape = self.image_shapes[self.current_progress]

  ###### Functions ######

  def AdaIN(self, x):
    # Normalize x[0] (image representation)
    mean = keras.backend.mean(x[0], axis = [1, 2], keepdims = True)
    std = keras.backend.std(x[0], axis = [1, 2], keepdims = True) + 1e-7
    y = (x[0] - mean) / std
    
    # Reshape scale and bias parameters
    pool_shape = [-1, 1, 1, y.shape[-1]]
    scale = keras.backend.reshape(x[1], pool_shape)
    bias = keras.backend.reshape(x[2], pool_shape)
    
    # Multiply by x[1] (GAMMA) and add x[2] (BETA)
    return y * scale + bias
  
  def mapping_block(self, latent_input):
    self.MAPPING_BLOCK_LEN = 3
    latent = layers.Dense(units=self.LATENT_SIZE, activation = 'relu')(latent_input)
    latent = layers.Dense(units=self.LATENT_SIZE, activation = 'relu')(latent)
    latent = layers.Dense(units=self.LATENT_SIZE, activation = 'relu')(latent)
    return latent

  def g_block(self, input_tensor, latent_vector, filters, upsamp=2):
    # Warning: this block is not a straight line!
    AdaIN = self.AdaIN

    gamma = layers.Dense(units=filters, bias_initializer = 'ones')(latent_vector)
    beta = layers.Dense(units=filters)(latent_vector)
    
    if upsamp > 1:
      out = layers.UpSampling2D(upsamp)(input_tensor)
    else:
      out = input_tensor
    out = layers.Conv2D(filters=filters, kernel_size=3, padding = 'same')(out)
    out = layers.Lambda(AdaIN)([out, gamma, beta])
    out = layers.Activation('relu')(out)
    
    return out
  
  def output_block(self, input_tensor, channels=4):
    self.OUTPUT_BLOCK_LEN = 1
    # make RGBA with values between 0 and 1
    x = layers.Conv2D(channels, 1, padding = 'same', activation = 'sigmoid')(input_tensor)
    return x

  ## build model (pre-progress)
  def build_model(self, strategy):
    AdaIN = self.AdaIN
    g_block = self.g_block
    output_block = self.output_block
    mapping_block = self.mapping_block

    with strategy.scope():
      # Latent input
      latent_input = layers.Input([self.LATENT_SIZE])

      # Map latent input
      latent = mapping_block(latent_input)

      # Reshape to 5x5x256
      x = layers.Dense(units=5*5*self.LATENT_SIZE, activation = 'relu')(latent)
      x = layers.Reshape([5, 5, self.LATENT_SIZE])(x)
      # Size: 5x5x256
      self.image_shape = (5, 5, 4)

      # output RGBA image
      image_output = output_block(x)

      # Make Model
      model = Model(inputs = latent_input, outputs = image_output)

      model.summary()

    return model
  
  ## GAN progressive training: add layers
  def progress_model(self, model, strategy):
    AdaIN = self.AdaIN
    g_block = self.g_block
    output_block = self.output_block
    current_progress = self.current_progress

    if current_progress >= self.MAX_PROGRESS:
      print("Maximum progress reached!")
      return model
    current_progress += 1

    with strategy.scope():
      # get layers from old model
      x = model.layers[-(self.OUTPUT_BLOCK_LEN+1)].output
      latent = model.layers[self.MAPPING_BLOCK_LEN].output            # indexing included input block (1)

      if current_progress == 1:
        x = g_block(x, latent, 256, 3)
        # Size: 15x15x256
        x = g_block(x, latent, 128, 1)
        # Size: 15x15x128
        self.image_shape = (15, 15, 4)

      elif current_progress == 2:
        x = g_block(x, latent, 64, 3)
        # Size: 45x45x64
        self.image_shape = (45, 45, 4)

      elif current_progress == 3:
        x = g_block(x, latent, 32)
        # Size: 90x90x32
        self.image_shape = (90, 90, 4)

      elif current_progress == 4:
        x = g_block(x, latent, 16)
        # Size: 180x180x16
        self.image_shape = (180, 180, 4)

      elif current_progress == 5:
        x = g_block(x, latent, 8)
        # Size: 360x360x8
        self.image_shape = (360, 360, 4)
      
      # output RGBA image
      image_output = output_block(x)

      # Make new model on top of the old model
      model = Model(inputs = model.input, outputs = image_output)

      model.summary()
    
    # return progressed model
    print("The model progressed to level " + str(current_progress))
    return model
  
