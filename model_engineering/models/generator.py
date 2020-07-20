########## Generator model ##########

###### Imports ######

import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.api._v2.keras import layers, Model

###### Constants ######

LATENT_SIZE = 256
IMAGE_SHAPE = (360, 360, 4)

###### Functions ######

def AdaIN(x):
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

def g_block(input_tensor, latent_vector, filters, upsamp=2):
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

def build_model(strategy):
  with strategy.scope():
    # Latent input
    latent_input = layers.Input([LATENT_SIZE])

    # Map latent input
    latent = layers.Dense(units=LATENT_SIZE, activation = 'relu')(latent_input)
    latent = layers.Dense(units=LATENT_SIZE, activation = 'relu')(latent)
    latent = layers.Dense(units=LATENT_SIZE, activation = 'relu')(latent)

    # Reshape to 3x3x64
    x = layers.Dense(units=5*5*LATENT_SIZE, activation = 'relu')(latent_input)
    x = layers.Reshape([5, 5, LATENT_SIZE])(x)

    # Size: 5x5x256
    x = g_block(x, latent, 256, 3)

    # Size: 15x15x256
    x = g_block(x, latent, 128, 1)

    # Size: 15x15x128
    x = g_block(x, latent, 64, 3)

    # Size: 45x45x64
    x = g_block(x, latent, 32)

    # Size: 90x90x32
    x = g_block(x, latent, 16)

    # Size: 180x180x16
    x = g_block(x, latent, 8)

    # Size: 360x360x8, make RGB with values between 0 and 1
    image_output = layers.Conv2D(4, 1, padding = 'same', activation = 'sigmoid')(x)

    # Make Model
    model = Model(inputs = latent_input, outputs = image_output)

    model.summary()

  return model

###### Execution ######

