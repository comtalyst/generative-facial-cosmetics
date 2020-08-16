########## Generator model ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
import os
import h5py
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model, models, backend
else:
  from tensorflow.keras import layers, Model, models, backend

###### Class Content ######
class Encoder:

  ###### Constants ######

  LATENT_SIZE = 256
  IMAGE_SHAPE = (360, 360, 0)
  model = None
  model_type = None

  ###### Constructor ######

  def __init__(self, strategy, model_type=None):
    self.model_type = 'custom'
    self.IMAGE_SHAPE = (self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 4)
    if model_type == None or type(model_type) != str:
      self.model = self.build_model(strategy)
    elif model_type.lower() in ['vgg', 'vgg16', 'vgg-16', 'vgg_16']:
      self.IMAGE_SHAPE = (self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 3)
      self.model_type = 'vgg16'
      self.model = self.build_model_vgg16(strategy)
    elif model_type.lower() in ['mirror', 'mir', 'discriminator', 'disc']:
      self.model_type = 'mirror'
      self.model = self.build_model_mirror(strategy)
    else:
      self.model = self.build_model(strategy)
  
  ###### Public Methods ######
  
  # save to SavedModel
  def save(self, epoch, strategy, dir = None):
    model = self.model

    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    fname = "encoder" + "-e_" + str(epoch)
    with strategy.scope():
      model.save(os.path.join(dir, fname))

  # either specify dir/fname or path (path takes priority)
  def load(self, strategy, dir = None, fname = None, path = None):
    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    if fname == None and path == None:
      raise ValueError("Please specify either fname or path")
    if path == None:
      path = os.path.join(dir, fname)
    with strategy.scope():
      self.model = models.load_model(path)
      self.model.summary()

  ###### Functions ######

  # conv block, borrowed from the discriminator
  def d_block(self, input_tensor, filters, reduce_times = 2):
    out = layers.Conv2D(filters, 3, padding = 'same')(input_tensor)
    out = layers.LeakyReLU(0.2)(out)
    if reduce_times > 1:
      out = layers.AveragePooling2D(reduce_times)(out)
    return out

  ## build model (custom), note that this model requires 4 channels
  def build_model(self, strategy):
    d_block = self.d_block
    with strategy.scope():
      input_layer = layers.Input(self.IMAGE_SHAPE)
      x = input_layer
      
      x = d_block(x, 64, 2)                       # 360x360x64, 180x180x64
      x = d_block(x, 128, 2)                      # 180x180x128, 90x90x128
      x = d_block(x, 256, 2)                      # 90x90x256, 45x45x256
      x = d_block(x, 512, 3)                      # 45x45x512, 15x15x512
      x = d_block(x, 512, 3)                      # 15x15x512, 5x5x512

      x = layers.Flatten()(x)                     # 5*5*512 = 12800
      x = layers.Dense(4096)(x)
      x = layers.Dense(4096)(x)
      x = layers.Dense(self.LATENT_SIZE)(x)

      model = Model(inputs=input_layer, outputs=x, name="custom-encoder")
      model.summary()

    return model
  
  # unused old version
  def build_model_mirror(self, strategy):
    d_block = self.d_block
    with strategy.scope():
      input_layer = layers.Input(self.IMAGE_SHAPE)
      x = input_layer
      
      x = d_block(x, 8)                       # 180x180x8
      x = d_block(x, 16)                      # 90x90x16
      x = d_block(x, 32)                      # 45x45x32
      x = d_block(x, 64, 3)                   # 15x15x64
      x = d_block(x, 128, 1)                  # 15x15x128
      x = d_block(x, 256, 3)                  # 5x5x256

      x = layers.Flatten()(x)                 # 5*5*256 = 6400
      x = layers.Dense(1024)(x)
      x = layers.Dense(1024)(x)
      x = layers.Dense(self.LATENT_SIZE)(x)

      model = Model(inputs=input_layer, outputs=x, name="mirrored-encoder")
      model.summary()

    return model

  ## build model
  def build_model_vgg16(self, strategy):
    with strategy.scope():
      input_layer = layers.Input(self.IMAGE_SHAPE)
      output = keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_layer).output
      output = layers.Flatten()(output)
      output = layers.Dense(1024)(output)
      output = layers.Dense(1024)(output)
      output = layers.Dense(self.LATENT_SIZE)(output)
      model = Model(inputs=input_layer, outputs=output, name="vgg16-encoder")
      
      model.summary()

    return model

