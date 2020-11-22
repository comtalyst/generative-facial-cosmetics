########## Generator model ##########

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
class Generator:

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

  def __init__(self, strategy, model_type=None, dropout=0):
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

  def progress(self, strategy, silent=False):
    if self.current_progress >= self.MAX_PROGRESS:
      print("Maximum progress reached!")
      return
    self.current_progress += 1
    # latent = 512
    if self.model_type == None or self.model_type == '512':
      self.model, self.model_fade = self.progress_model(self.model, self.current_progress, strategy, silent)
    # latent = 256
    elif self.model_type == '256':
      self.model, self.model_fade = self.progress_model_256(self.model, self.current_progress, strategy, silent)
    
  ## save to SavedModel (does not include fade)
  def save(self, epoch, strategy, dir = None):
    model = self.model

    if dir == None:
      dir = os.path.join(DIR_OUTPUT, os.path.join('saved_models', 'current'))
    fname = "generator" + "-p_" + str(self.current_progress) + "-e_" + str(epoch)
    with strategy.scope():
      model.save(os.path.join(dir, fname))

  ## either specify dir/fname or path (path takes priority)
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

  ## set alpha value for fading-in model
  def setAlpha(self, new_alpha):
    found = False
    for layer in self.model_fade.layers:
      if isinstance(layer, WeightedSum):
        backend.set_value(layer.alpha, new_alpha)
        found = True
    if not found:
      raise RuntimeError("Cannot set alpha on undefined model_fade; make sure this is not a first-gen model")

  ## get model with injection options based on current self.model
  def get_injectible_model(self, strategy):
    '''
    There is a possible alternative method where we reconstruct the graph, attach new inputs
    then topologically rebuild the model. That method will save more memory, but at the cost of
    more complex implementation.
    '''
    ## construct a new model with the same architecture
    # latent = 512
    if self.model_type == None or self.model_type == '512':
      injectible_model = self.build_model(strategy, True)
      for i in range(self.current_progress):
        injectible_model, _ = self.progress_model(injectible_model, i+1, strategy, True, True)
    # latent = 256
    elif self.model_type == '256':
      injectible_model = self.build_model_256(strategy, True)
      for i in range(self.current_progress):
        injectible_model, _ = self.progress_model_256(injectible_model, i+1, strategy, True, True)

    ## transfer weights 
    old_pointer_idx = 0
    for layer in injectible_model.layers:
      if len(layer.get_weights()) == 0:       # input layers do not have weights and will fall into this trap
        continue
      while len(self.model.layers[old_pointer_idx].get_weights()) == 0: # this one too
        old_pointer_idx += 1
      old_layer = self.model.layers[old_pointer_idx]
      layer.set_weights(old_layer.get_weights())
      old_pointer_idx += 1
    
    injectible_model.summary()

    return injectible_model

  ###### Blocks ######

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
    self.MAPPING_BLOCK_LEN = 6
    latent = layers.Dense(units=self.LATENT_SIZE, name='mapping_dense0')(latent_input)
    latent = layers.LeakyReLU(0.2, name='mapping_activ0')(latent)
    latent = layers.Dense(units=self.LATENT_SIZE, name='mapping_dense1')(latent)
    latent = layers.LeakyReLU(0.2, name='mapping_activ1')(latent)
    latent = layers.Dense(units=self.LATENT_SIZE, name='mapping_dense2')(latent)
    latent = layers.LeakyReLU(0.2, name='mapping_activ2')(latent)
    return latent

  def g_block(self, input_tensor, latent_vector, filters, upsamp, name=None):
    # Warning: this block is not a straight line!
    AdaIN = self.AdaIN

    if name == None:
      name = str(filters)

    gamma = layers.Dense(units=filters, bias_initializer = 'ones', name=name+'_latent_gamma')(latent_vector)
    beta = layers.Dense(units=filters, name=name+'_latent_beta')(latent_vector)
    
    if upsamp > 1:
      out = layers.UpSampling2D(upsamp, name=name+'_upsamp')(input_tensor)
    else:
      out = input_tensor
    out = layers.Conv2D(filters=filters, kernel_size=3, padding = 'same', name=name+'_conv')(out)
    out = layers.Lambda(AdaIN)([out, gamma, beta])
    out = layers.LeakyReLU(0.2, name=name+'_activ')(out)
    
    return out
  
  def output_block(self, input_tensor, channels=4):
    self.OUTPUT_BLOCK_LEN = 1
    # make RGBA with values between 0 and 1
    x = layers.Conv2D(channels, 1, padding = 'same', activation = 'sigmoid', name='output_'+str(input_tensor.shape[1]) + '_conv')(input_tensor)
    return x

  ###### Utils ######

  ## get get new latent from input and map them
  def get_new_latent(self, model, name=None):
    if name == None:
      new_input = layers.Input([self.LATENT_SIZE])
    else:
      new_input = layers.Input([self.LATENT_SIZE], name=name+'_input')
    new_latent = new_input
    for i in range(1, self.MAPPING_BLOCK_LEN+1):
      new_latent = model.layers[i](new_latent)
    return new_latent, new_input

  ###### Modeling ######

  ## build model (pre-progress)
  def build_model(self, strategy, silent=False):
    AdaIN = self.AdaIN
    g_block = self.g_block
    output_block = self.output_block
    mapping_block = self.mapping_block
    dropout = self.dropout

    with strategy.scope():
      # Latent input
      latent_input = layers.Input([self.LATENT_SIZE], name='root_input')

      # Map latent input
      latent = mapping_block(latent_input)

      # Reshape to 5x5x512
      x = layers.Dense(units=5*5*self.LATENT_SIZE, name='aftermap_dense')(latent)
      x = layers.LeakyReLU(0.2, name='aftermap_activ')(x)
      x = layers.Dropout(dropout)(x)
      x = layers.Reshape([5, 5, self.LATENT_SIZE], name='aftermap_reshape')(x)
      # Size: 5x5x512
      self.image_shape = (5, 5, 4)

      # output RGBA image
      image_output = output_block(x)

      # Make Model
      model = Model(inputs = latent_input, outputs = image_output, name='generator-512-0')

      if not silent:
        model.summary()

    return model
  
  ## GAN progressive training: add layers
  def progress_model(self, model, next_progress, strategy, silent=False, injectible=False):
    AdaIN = self.AdaIN
    g_block = self.g_block
    output_block = self.output_block

    with strategy.scope():
      # get layers from old model
      old_output = model.layers[-(self.OUTPUT_BLOCK_LEN+1)].output
      old_latent = model.layers[self.MAPPING_BLOCK_LEN].output            # indexing included input block (1)
      new_inputs = model.inputs

      if next_progress == 1:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '256')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 256, 3)           # Size: 15x15x256
        '''
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '256')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(x, latent, 256, 1)                    # Size: 15x15x256
        '''

        ## upsampling only path
        y = layers.UpSampling2D(3)(old_output)            # Size: 15x15x512
        self.image_shape = (15, 15, 4)

      elif next_progress == 2:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '128')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 128, 3)            # Size: 45x45x128

        ## upsampling only path
        y = layers.UpSampling2D(3)(old_output)            # Size: 45x45x256
        self.image_shape = (45, 45, 4)

      elif next_progress == 3:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '64')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 64, 2)               # Size: 90x90x64

        ## upsampling only path
        y = layers.UpSampling2D(2)(old_output)            # Size: 90x90x128
        self.image_shape = (90, 90, 4)

      elif next_progress == 4:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '32')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 32, 2)               # Size: 180x180x32

        ## upsampling only path
        y = layers.UpSampling2D(2)(old_output)            # Size: 180x180x64
        self.image_shape = (180, 180, 4)

      elif next_progress == 5:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '16')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 16, 2)                # Size: 360x360x16

        ## upsampling only path
        y = layers.UpSampling2D(2)(old_output)            # Size: 360x360x32
        self.image_shape = (360, 360, 4)
      else:
        print('Unknown progress level')
        return None
      
      # output RGBA image
      full_output = output_block(x)                       # Size: ???x???x4

      # use old (to be trashsed) output block for fading model
      samp_output = y
      for i in range(len(model.layers)-self.OUTPUT_BLOCK_LEN, len(model.layers)):
        samp_output = model.layers[i](samp_output)        # Size: ???x???x4
      fade_output = WeightedSum()([samp_output, full_output])    # transition from upsampling only to full

      # Make new model on top of the old model
      if injectible:
        suf = '-i'
      else:
        suf = ''
      model_full = Model(inputs = new_inputs, outputs = full_output, name='generator-512-'+str(next_progress)+suf)
      model_fade = Model(inputs = new_inputs, outputs = fade_output, name='generator-512-'+str(next_progress)+'-fade'+suf)

      if not silent:
        model_full.summary()
        print("The model progressed to level " + str(next_progress))
    
    # return progressed model
    return model_full, model_fade

  def g_block_256(self, input_tensor, latent_vector, filters, upsamp, name=None):
    # Warning: this block is not a straight line!
    AdaIN = self.AdaIN

    if name == None:
      name = str(filters)

    gamma = layers.Dense(units=filters, bias_initializer = 'ones', name=name+'_latent_gamma')(latent_vector)
    beta = layers.Dense(units=filters, name=name+'_latent_beta')(latent_vector)
    
    if upsamp > 1:
      out = layers.UpSampling2D(upsamp, name=name+'_upsamp')(input_tensor)
    else:
      out = input_tensor
    out = layers.Conv2D(filters=filters, kernel_size=3, padding = 'same', name=name+'_conv')(out)
    out = layers.Lambda(AdaIN)([out, gamma, beta])
    out = layers.Activation('relu', name=name+'_activ')(out)
    
    return out

  def mapping_block_256(self, latent_input):
    latent = layers.Dense(units=self.LATENT_SIZE, activation = 'relu')(latent_input)
    latent = layers.Dense(units=self.LATENT_SIZE, activation = 'relu')(latent)
    latent = layers.Dense(units=self.LATENT_SIZE, activation = 'relu')(latent)
    return latent

  ## build model (latent=256, pre-progress)
  def build_model_256(self, strategy, silent=False):
    AdaIN = self.AdaIN
    g_block = self.g_block_256
    output_block = self.output_block
    mapping_block = self.mapping_block_256
    self.MAPPING_BLOCK_LEN = 3

    with strategy.scope():
      # Latent input
      latent_input = layers.Input([self.LATENT_SIZE])

      # Map latent input
      latent = mapping_block(latent_input)

      # Reshape to 5x5x256
      x = layers.Dense(units=5*5*self.LATENT_SIZE)(latent)
      x = layers.LeakyReLU(0.2)(x)
      x = layers.Reshape([5, 5, self.LATENT_SIZE])(x)
      # Size: 5x5x256
      self.image_shape = (5, 5, 4)

      # output RGBA image
      image_output = output_block(x)

      # Make Model
      model = Model(inputs = latent_input, outputs = image_output, name='generator-256-0')

      if not silent:
        model.summary()

    return model
  
  ## GAN progressive training: add layers (latent=256)  
  def progress_model_256(self, model, next_progress, strategy, silent=False, injectible=False):
    '''
    Fading Generator: 
      Full: Old Pre-Output Block --> Upsampling --> Conv/AdaIN --> New Output Block --> COMBINE THE RESULTING IMAGES
      Samp: Old Pre-Output Block --> Upsampling --> Old Output Block ----------------->
    '''
    AdaIN = self.AdaIN
    g_block = self.g_block_256
    output_block = self.output_block

    with strategy.scope():
      # get layers from old model
      old_output = model.layers[-(self.OUTPUT_BLOCK_LEN+1)].output
      old_latent = model.layers[self.MAPPING_BLOCK_LEN].output            # indexing included input block (1)
      new_inputs = model.inputs

      if next_progress == 1:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '256')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 256, 3)           # Size: 15x15x256

        if injectible:
          new_latent, new_input = self.get_new_latent(model, '128')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(x, latent, 128, 1)                    # Size: 15x15x128

        ## upsampling only path
        y = layers.UpSampling2D(3)(old_output)            # Size: 15x15x256
        self.image_shape = (15, 15, 4)

      elif next_progress == 2:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '64')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 64, 3)            # Size: 45x45x64

        ## upsampling only path
        y = layers.UpSampling2D(3)(old_output)            # Size: 45x45x128
        self.image_shape = (45, 45, 4)

      elif next_progress == 3:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '32')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 32, 2)               # Size: 90x90x32

        ## upsampling only path
        y = layers.UpSampling2D(2)(old_output)            # Size: 90x90x64
        self.image_shape = (90, 90, 4)

      elif next_progress == 4:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '16')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 16, 2)               # Size: 180x180x16

        ## upsampling only path
        y = layers.UpSampling2D(2)(old_output)            # Size: 180x180x32
        self.image_shape = (180, 180, 4)

      elif next_progress == 5:
        ## main (upsampling + AdaIN) path
        if injectible:
          new_latent, new_input = self.get_new_latent(model, '8')
          new_inputs.append(new_input)
          latent = new_latent
        else:
          latent = old_latent
        x = g_block(old_output, latent, 8, 2)                # Size: 360x360x8

        ## upsampling only path
        y = layers.UpSampling2D(2)(old_output)            # Size: 360x360x16
        self.image_shape = (360, 360, 4)
      else:
        print('Unknown progress level')
        return None
      
      # output RGBA image
      full_output = output_block(x)                       # Size: ???x???x4

      # use old (to be trashsed) output block for fading model
      samp_output = y
      for i in range(len(model.layers)-self.OUTPUT_BLOCK_LEN, len(model.layers)):
        samp_output = model.layers[i](samp_output)        # Size: ???x???x4
      fade_output = WeightedSum()([samp_output, full_output])    # transition from upsampling only to full

      # Make new model on top of the old model
      if injectible:
        suf = '-i'
      else:
        suf = ''
      model_full = Model(inputs = new_inputs, outputs = full_output, name='generator-256-'+str(next_progress)+suf)
      model_fade = Model(inputs = new_inputs, outputs = fade_output, name='generator-256-'+str(next_progress)+'-fade'+suf)

      if not silent:
        model_full.summary()
        print("The model progressed to level " + str(next_progress))
    
    # return progressed model
    return model_full, model_fade
 
    ## get model that is injectible by other latent vectors

  ## currently unused
  def get_injectible_layers(self):
    injectibles = list()
    for layer in self.model.layers:
      try:
        if list(layer.input.shape) == [None, self.LATENT_SIZE] and len(layer.get_weights()) != 0:
            injectibles.append(layer)
      except:
        pass
    injectibles = injectibles[self.MAPPING_BLOCK_LEN+1:]
    return injectibles
