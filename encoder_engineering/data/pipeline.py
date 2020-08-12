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
GENERATOR = None

###### Functions ######

### use trained stylegan to generate image from latent space
def gen_img(noise):
  if GENERATOR == None:
    raise RuntimeError("The generator is not defined or not found")
  return (GENERATOR.model(tf.expand_dims(noise, 0), training=False)[0], noise)

### preprocess input
def preprocess(image, noise):
  '''
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if image[i, j, 3] < 128:
        image[i, j] = (0, 0, 0, 0)
  '''
  # blackout the transparents and reduce channels from 4 to 3
  mask = tf.dtypes.cast((image[:, :, 3] >= 128), tf.float32)
  image = tf.math.multiply(image, tf.expand_dims(mask, 2))
  image = image[:, :, :3]
  return (tf.keras.applications.vgg16.preprocess_input(image), noise)

### return "list" of (latent, generator output)
def load_dataset(n, batch=False):
  option_no_order = tf.data.Options()
  option_no_order.experimental_deterministic = False
  
  rand_latents = tf.random.normal([n, LATENT_SIZE])
  dataset = tf.data.Dataset.from_tensor_slices(rand_latents)
  dataset = dataset.with_options(option_no_order)
  dataset = dataset.map(gen_img, num_parallel_calls=AUTO) 
  dataset = dataset.map(preprocess, num_parallel_calls=AUTO) 
  if batch:
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) 
  return dataset

### just call this ez func
def get_dataset(generator, strategy, n_train=BATCH_SIZE*64, n_valid=BATCH_SIZE*64, batch=False, image_size=DEFAULT_IMAGE_SIZE):
  global IMAGE_SIZE
  IMAGE_SIZE = image_size

  global GENERATOR 
  GENERATOR = generator

  return load_dataset(n_train, batch), load_dataset(n_valid, batch)

###### Execution ######

