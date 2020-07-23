########## Discriminator model ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model, models
else:
  from tensorflow.keras import layers, Model

###### Class Content ######
class Discriminator:

  ###### Constants ######

  LATENT_SIZE = 256
  IMAGE_SHAPE = (360, 360, 4)
  MAX_PROGRESS = 5
  model = None

  current_progress = 0              # for GAN progressive training

  ###### Constructor ######

  def __init__(self, strategy):
    self.model = self.build_model(strategy)

  ###### Public Methods ######

  def progress(self, strategy):
    self.model = self.progress_model(self.model, strategy)
  
  def save(self, epoch, dir = None):
    model = self.model

    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    fname = "discriminator" + "-p_" + str(current_progress) + "-e_" + str(epoch) + ".h5"
    model.save(fname)
  
  # either specify dir/fname or path (path takes priority)
  def load(self, its_progress, dir = None, fname = None, path = None):
    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    if fname == None and path == None:
      raise ValueError("Please specify either fname or path")
    if path == None:
      path = os.path.join(dir, fname)
    current_progress = its_progress
    self.model = models.load_model(path)

  ###### Functions ######

  def d_block(self, input_tensor, filters, reduce_times = 2):
    out = layers.Conv2D(filters, 3, padding = 'same')(input_tensor)
    out = layers.LeakyReLU(0.2)(out)
    if reduce_times > 1:
      out = layers.AveragePooling2D(reduce_times)(out)
    return out
  
  def output_block(self, input_tensor, units=512):
    self.OUTPUT_BLOCK_LEN = 4

    x = layers.Conv2D(units, 3, padding = 'same')(input_tensor)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    # 1-dimensional Neural Network
    return layers.Dense(1, name="output_prob")(x)

  def build_model(self, strategy):
    d_block = self.d_block
    output_block = self.output_block

    with strategy.scope():
      # Image input
      image_input = layers.Input(self.IMAGE_SHAPE, name="input_image")
      # Size: 360x360x4

      # convert to prob for decision
      class_output = output_block(image_input)

      # Make Model
      model = Model(inputs = image_input, outputs = class_output)

      model.summary()

    return model
  
  ## GAN progressive training: add layers
  def progress_model(self, model, strategy):
    d_block = self.d_block
    output_block = self.output_block
    current_progress = self.current_progress

    if current_progress >= self.MAX_PROGRESS:
      print("Maximum progress reached!")
      return model
    current_progress += 1

    with strategy.scope():
      # get last layer before output (before reducing to 1 neuron)
      x = model.layers[-(self.OUTPUT_BLOCK_LEN+1)].output

      if current_progress == 1:
        x = d_block(x, 8)
        # Size: 180x180x8

      elif current_progress == 2:
        x = d_block(x, 16)
        # Size: 90x90x16

      elif current_progress == 3:
        x = d_block(x, 32)
        # Size: 45x45x32

      elif current_progress == 4:
        x = d_block(x, 64, 3)
        # Size: 15x15x64
        x = d_block(x, 128, 1)
        # Size: 15x15x128

      elif current_progress == 5:
        x = d_block(x, 256, 3)
        # Size: 5x5x256
      
      # convert to prob for decision
      class_output = output_block(x)

      # Make new model on top of the old model
      model = Model(inputs = model.input, outputs = class_output)

      model.summary()
    
    # return progressed model
    print("The model progressed to level " + str(current_progress))
    return model


