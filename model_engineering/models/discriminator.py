########## Discriminator model ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
import os
import h5py
from models.custom_layers.WeightedSum import WeightedSum
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model, models, backend
else:
  from tensorflow.keras import layers, Model, models, backend

###### Class Content ######
class Discriminator:

  ###### Constants ######

  LATENT_SIZE = 512
  FINAL_IMAGE_SHAPE = (360, 360, 4)
  MAX_PROGRESS = 5
  image_shapes = {
    0: (5, 5, 4),
    1: (15, 15, 4),
    2: (45, 45, 4),
    3: (90, 90, 4),
    4: (180, 180, 4),
    5: (360, 360, 4)
  }
  current_progress = 0              # for GAN progressive training

  ## to-be-defined
  model = None
  model_type = None
  model_fade = None         # fading-to-grown model
  dropout = None

  ###### Constructor ######

  def __init__(self, strategy, model_type=None, dropout=0.2):
    self.dropout = dropout

    # latent = 512
    if model_type == None or model_type == '512':
      model_type = '512'
      self.model = self.build_model(strategy)
    # latent = 256
    elif model_type == '256':
      self.model = self.build_model_256(strategy)
    else:
      raise ValueError('Unknown model_type: ' + str(model_type))
    
    self.model_type = model_type
    self.model_fade = self.model    # placeholder for checkpointing

  ###### Public Methods ######

  def progress(self, strategy):   
    # latent = 512
    if self.model_type == None or self.model_type == '512':
      self.model, self.model_fade = self.progress_model(self.model, strategy)
    # latent = 256
    elif self.model_type == '256':
      self.model, self.model_fade = self.progress_model_256(self.model, strategy)
    
    self.current_progress += 1
  
  # save to SavedModel (does not include fade)
  def save(self, epoch, strategy, dir = None):
    model = self.model

    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    fname = "discriminator" + "-p_" + str(self.current_progress) + "-e_" + str(epoch)
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

  # set alpha value for fading-in model
  def setAlpha(self, new_alpha):
    found = False
    for layer in self.model_fade.layers:
      if isinstance(layer, WeightedSum):
        backend.set_value(layer.alpha, new_alpha)
        found = True
    if not found:
      raise RuntimeError("Cannot set alpha on undefined model_fade; make sure this is not a first-gen model")
  
  ###### Functions ######

  def d_block(self, input_tensor, filters, reduce_times = 2, name=None):
    if name == None:
      name = str(filters)

    out = layers.Conv2D(filters, 3, padding = 'same', name=name+'_conv')(input_tensor)
    out = layers.LeakyReLU(0.2, name=name+'_activ')(out)
    if reduce_times > 1:
      out = layers.AveragePooling2D(reduce_times, name=name+'_downsamp')(out)
    return out
  
  def input_block(self, shape, filters):
    self.INPUT_BLOCK_LEN = 3
    # Image input
    image_input = layers.Input(shape, name='input_'+str(shape[0]))
    # expand filters without affecting other variables (this layer will be deleted in the next generation)
    x = layers.Conv2D(filters, 1, padding = 'same', name='input_'+str(shape[0])+'_conv')(image_input)
    x = layers.LeakyReLU(0.2, name='input_'+str(shape[0])+'_activ')(x)
    return x, image_input
    
  def build_model(self, strategy):
    d_block = self.d_block
    input_block = self.input_block
    dropout = self.dropout

    with strategy.scope():
      # Image input
      self.image_shape = (5, 5, 4)
      x, image_input = input_block((5, 5, 4), 512)        # Size: 5x5x512

      # convert to prob for decision using 1-dimensional Neural Network
      x = d_block(x, 2048, reduce_times = 1)
      x = layers.Flatten()(x)
      x = layers.Dropout(dropout)(x)
      #x = layers.Dense(2048, name='prefinal_dense')(x)
      #x = layers.LeakyReLU(0.2, name='prefinal_activ')(x)
      x = layers.Dense(1, name='final_dense')(x)

      # Make Model
      model = Model(inputs = image_input, outputs = x, name='discriminator-512-0')

      model.summary()

    return model
  
  ## GAN progressive training: add layers
  def progress_model(self, model, strategy):
    d_block = self.d_block
    input_block = self.input_block
    current_progress = self.current_progress      # note that the generator currently use a newer system on progress referencing

    if current_progress >= self.MAX_PROGRESS:
      print("Maximum progress reached!")
      return model
    current_progress += 1

    with strategy.scope():
      if current_progress == 1:
        self.image_shape = (15, 15, 4)
        x, image_input = input_block((15, 15, 4), 128)     # Size: 15x15x128
        ## main (Conv + downsampling) path
        x = d_block(x, 256, 1)                            # Size: 15x15x256
        x = d_block(x, 512, 3)                            # Size: 5x5x512
        # should now be match with y's preferred input shape
        ## downsampling only path
        y = layers.AveragePooling2D(3, name='512_fade_downsamp')(image_input)       # Size: 5x5x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 5x5x512

      elif current_progress == 2:
        self.image_shape = (45, 45, 4)
        x, image_input = input_block((45, 45, 4), 64)     # Size: 45x45x64
        ## main (Conv + downsampling) path
        x = d_block(x, 128, 3)                             # Size: 15x15x128
        ## downsampling only path
        y = layers.AveragePooling2D(3, name='128_fade_downsamp')(image_input)       # Size: 15x15x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 15x15x128

      elif current_progress == 3:
        self.image_shape = (90, 90, 4)
        x, image_input = input_block((90, 90, 4), 32)     # Size: 90x90x32
        ## main (Conv + downsampling) path
        x = d_block(x, 64, 2)                             # Size: 45x45x64
        ## downsampling only path
        y = layers.AveragePooling2D(2, name='64_fade_downsamp')(image_input)       # Size: 45x45x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 45x45x64

      elif current_progress == 4:
        self.image_shape = (180, 180, 4)
        x, image_input = input_block((180, 180, 4), 16)    # Size: 180x180x16
        ## main (Conv + downsampling) path
        x = d_block(x, 32, 2)                             # Size: 90x90x32
        ## downsampling only path
        y = layers.AveragePooling2D(2, name='32_fade_downsamp')(image_input)       # Size: 90x90x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 90x90x32

      elif current_progress == 5:
        self.image_shape = (360, 360, 4)
        x, image_input = input_block((360, 360, 4), 8)    # Size: 360x360x8
        ## main (Conv + downsampling) path
        x = d_block(image_input, 16)                       # Size: 180x180x16
        ## downsampling only path
        y = layers.AveragePooling2D(2, name='16_fade_downsamp')(image_input)       # Size: 180x180x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 180x180x16
      
      # merge for fading model
      merged = WeightedSum()([y, x])

      # continue on with the model
      for i in range(self.INPUT_BLOCK_LEN, len(model.layers)):
        x = model.layers[i](x)
        merged = model.layers[i](merged)

      # Make new model on top of the old model
      model_full = Model(inputs = image_input, outputs = x, name='discriminator-512-'+str(current_progress))
      model_fade = Model(inputs = image_input, outputs = merged, name='discriminator-512-'+str(current_progress)+'-fade')

      model_full.summary()
    
    # return progressed model
    print("The model progressed to level " + str(current_progress))
    return model_full, model_fade

  def build_model_256(self, strategy):
    d_block = self.d_block
    input_block = self.input_block

    with strategy.scope():
      # Image input
      self.image_shape = (5, 5, 4)
      x, image_input = input_block((5, 5, 4), 256)        # Size: 5x5x256

      # convert to prob for decision using 1-dimensional Neural Network
      x = d_block(x, 2048, reduce_times = 1)
      x = layers.Flatten()(x)
      x = layers.Dense(1, name='final_dense')(x)

      # Make Model
      model = Model(inputs = image_input, outputs = x, name='discriminator-256-0')

      model.summary()

    return model
  
  ## GAN progressive training: add layers
  def progress_model_256(self, model, strategy):
    '''
    Fading Discriminator: 
      Full: New Input --> New Input Block --> Conv --> Downsampling --> COMBINE --> Old Layers
      Samp: New Input --> Downsampling --> Old Input Block ----------->
    '''
    d_block = self.d_block
    input_block = self.input_block
    current_progress = self.current_progress

    if current_progress >= self.MAX_PROGRESS:
      print("Maximum progress reached!")
      return model
    current_progress += 1

    with strategy.scope():
      if current_progress == 1:
        self.image_shape = (15, 15, 4)
        x, image_input = input_block((15, 15, 4), 64)     # Size: 15x15x64
        ## main (Conv + downsampling) path
        x = d_block(x, 128, 1)                            # Size: 15x15x128
        x = d_block(x, 256, 3)                            # Size: 5x5x256
        # should now be match with y's preferred input shape
        ## downsampling only path
        y = layers.AveragePooling2D(3, name='256_fade_downsamp')(image_input)       # Size: 5x5x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 5x5x256

      elif current_progress == 2:
        self.image_shape = (45, 45, 4)
        x, image_input = input_block((45, 45, 4), 32)     # Size: 45x45x32
        ## main (Conv + downsampling) path
        x = d_block(x, 64, 3)                             # Size: 15x15x64
        ## downsampling only path
        y = layers.AveragePooling2D(3, name='64_fade_downsamp')(image_input)       # Size: 15x15x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 15x15x64

      elif current_progress == 3:
        self.image_shape = (90, 90, 4)
        x, image_input = input_block((90, 90, 4), 16)     # Size: 90x90x16
        ## main (Conv + downsampling) path
        x = d_block(x, 32, 2)                             # Size: 45x45x32
        ## downsampling only path
        y = layers.AveragePooling2D(2, name='32_fade_downsamp')(image_input)       # Size: 45x45x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 45x45x32

      elif current_progress == 4:
        self.image_shape = (180, 180, 4)
        x, image_input = input_block((180, 180, 4), 8)    # Size: 180x180x8
        ## main (Conv + downsampling) path
        x = d_block(x, 16, 2)                             # Size: 90x90x16
        ## downsampling only path
        y = layers.AveragePooling2D(2, name='16_fade_downsamp')(image_input)       # Size: 90x90x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 90x90x16

      elif current_progress == 5:
        self.image_shape = (360, 360, 4)
        x, image_input = input_block((360, 360, 4), 4)    # Size: 360x360x4
        ## main (Conv + downsampling) path
        x = d_block(image_input, 8)                       # Size: 180x180x8
        ## downsampling only path
        y = layers.AveragePooling2D(2, name='8_fade_downsamp')(image_input)       # Size: 180x180x4
        # use old (to be trashsed) input block
        for i in range(1, self.INPUT_BLOCK_LEN):
          y = model.layers[i](y)                          # Size: 180x180x8
      
      # merge for fading model
      merged = WeightedSum()([y, x])

      # continue on with the model
      for i in range(self.INPUT_BLOCK_LEN, len(model.layers)):
        x = model.layers[i](x)
        merged = model.layers[i](merged)

      # Make new model on top of the old model
      model_full = Model(inputs = image_input, outputs = x, name='discriminator-256-'+str(current_progress))
      model_fade = Model(inputs = image_input, outputs = merged, name='discriminator-256-'+str(current_progress)+'-fade')

      model_full.summary()
    
    # return progressed model
    print("The model progressed to level " + str(current_progress))
    return model_full, model_fade

