########## Input Data Pipeline ##########

###### Imports ######

from config import *
import os
import tensorflow as tf
from technical.accelerators import BATCH_SIZE as bs
from PIL import Image
import json
from models.generator import Generator
import numpy as np

###### Constants ######

AUTO = tf.data.experimental.AUTOTUNE
DEFAULT_IMAGE_SIZE = [360, 360]
LATENT_SIZE = 256
BATCH_SIZE = bs
## to be defined
GENERATOR = None
IMAGE_SIZE = None

###### Functions ######

### use trained stylegan to generate image from latent space
def gen_img(noise):
  if GENERATOR == None:
    raise RuntimeError("The generator is not defined or not found")
  return (GENERATOR.model(tf.expand_dims(noise, 0), training=False)[0], noise)

### preprocess input (vgg-16)
def preprocess_vgg16(image, noise):
  # blackout the transparents and reduce channels from 4 to 3
  mask = tf.dtypes.cast((image[:, :, 3] >= 0.5), tf.float32)
  image = tf.math.multiply(image, tf.expand_dims(mask, 2))
  image = image[:, :, :3]
  image = tf.math.multiply(image, 255)        # from [0, 1] to [0, 255] for vgg16 preprocessing
  return (tf.keras.applications.vgg16.preprocess_input(image), noise)

### preprocess input (customized)
def preprocess(image, noise):
  return (image, noise)

### return "list" of (latent, generator output)
def load_dataset(model_type, n, strategy, batch=False):
  option_no_order = tf.data.Options()
  option_no_order.experimental_deterministic = False
  
  rand_latents = tf.random.normal([n, LATENT_SIZE])
  dataset = tf.data.Dataset.from_tensor_slices(rand_latents)
  dataset = dataset.with_options(option_no_order)
  with strategy.scope():
    dataset = dataset.map(gen_img, num_parallel_calls=AUTO) 

  ## preprocess by specified model type
  if model_type == None or type(model_type) != str:
    dataset = dataset.map(preprocess, num_parallel_calls=AUTO) 
  elif model_type.lower() in ['vgg', 'vgg16', 'vgg-16', 'vgg_16']:
    dataset = dataset.map(preprocess_vgg16, num_parallel_calls=AUTO) 
  else:
    dataset = dataset.map(preprocess, num_parallel_calls=AUTO) 
  
  ## reshuffling after each iteration
  dataset.shuffle(n, reshuffle_each_iteration=True)

  ## batching
  if batch:
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) 
  return dataset

### just call this ez func
def get_dataset(generator, strategy, model_type=None, n_train=BATCH_SIZE*8, n_valid=BATCH_SIZE*8, batch=False, image_size=DEFAULT_IMAGE_SIZE):
  global IMAGE_SIZE
  IMAGE_SIZE = image_size

  global GENERATOR 
  GENERATOR = generator     # use global since we are going to use dataset.map()

  return load_dataset(model_type, n_train, strategy, batch), load_dataset(model_type, n_valid, strategy, batch)

###### Execution ######

