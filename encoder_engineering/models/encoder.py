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
  IMAGE_SHAPE = (360, 360, 3)
  model = None

  ###### Constructor ######

  def __init__(self, strategy):
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

  ## build model (pre-progress)
  def build_model(self, strategy):
    with strategy.scope():
      input_layer = layers.Input(self.IMAGE_SHAPE)
      output = keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_layer).output
      output = layers.Flatten()(output)
      output = layers.Dense(1024)(output)
      output = layers.Dense(1024)(output)
      output = layers.Dense(self.LATENT_SIZE)(output)
      model = Model(inputs=input_layer, outputs=output)
      
      model.summary()

    return model

