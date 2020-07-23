########## Generator model ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model
else:
  from tensorflow.keras import layers, Model

###### Class Content ######
class Generator:

  ###### Constants ######

  LATENT_SIZE = 128
  IMAGE_SHAPE = (15, 15, 4)
  model = None

  ###### Constructor ######

  def __init__(self, strategy):
    self.model = self.build_model(strategy)

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

  def g_block(self, input_tensor, latent_vector, filters, upsamp=2):
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

  def build_model(self, strategy):
    AdaIN = self.AdaIN
    g_block = self.g_block

    with strategy.scope():
      # Latent input
      latent_input = layers.Input([self.LATENT_SIZE], name="input_latent")

      # Map latent input
      latent = layers.Dense(units=self.LATENT_SIZE, activation = 'relu')(latent_input)
      latent = layers.Dense(units=self.LATENT_SIZE, activation = 'relu')(latent)
      latent = layers.Dense(units=self.LATENT_SIZE, activation = 'relu')(latent)

      # Reshape to 5x5x128
      x = layers.Dense(units=5*5*self.LATENT_SIZE, activation = 'relu')(latent)
      x = layers.Reshape([5, 5, self.LATENT_SIZE])(x)

      # Size: 5x5x128
      x = g_block(x, latent, 64, 3)

      # Size: 15x15x64
      x = g_block(x, latent, 4, 1)

      # Size: 15x15x4
      # make RGB with values between 0 and 1
      image_output = layers.Conv2D(4, 1, padding = 'same', activation = 'sigmoid', name="output_latent")(x)

      # Make Model
      model = Model(inputs = latent_input, outputs = image_output)

      model.summary()

    return model
  