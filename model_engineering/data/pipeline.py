########## Input Data Pipeline ##########

###### Imports ######

from config import *
import os
import tensorflow as tf
from technical.accelerators import BATCH_SIZE as bs

###### Constants ######

AUTO = tf.data.experimental.AUTOTUNE
ORIG_IMAGE_SIZE = [360, 360]
IMAGE_SIZE = [360, 360]
BATCH_SIZE = bs


###### Functions ######

### convert tfrecord of PNGs to image tensors
def read_tfrecord(example):
  features = {
    "image": tf.io.FixedLenFeature([], tf.string),        # tf.string means bytestring
    "id": tf.io.FixedLenFeature([], tf.int64)             # probably unused, for now
  }
  example = tf.io.parse_example(example, features)
  images = tf.image.decode_png(example['image'], channels=3)
  images = tf.cast(images, tf.float32) / 255.0              # convert image to floats in [0, 1] range
  images = tf.image.resize(images, IMAGE_SIZE)              # explicit size will be needed for TPU
  return images
  
### return "list" of image tensors from specified TFRecords
def load_dataset(filenames, shuffle=False, batch=False):
  # read from TFRecords. For optimal performance, read from multiple
  # TFRecord files at once and set the option experimental_deterministic = False
  # to allow order-altering optimizations.

  option_no_order = tf.data.Options()
  option_no_order.experimental_deterministic = False

  dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
  dataset = dataset.with_options(option_no_order)
  dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)   # dataset from tfrecords is now in a parallel (normal/good) format
  if shuffle:
    dataset = dataset.shuffle(len(filenames))
  if batch:
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) 
  return dataset

### get filenames
def get_tfrec_names(dir):
  PATH_PATTERN = os.path.join(dir, '*.tfrec')
  filenames = tf.io.gfile.glob(PATH_PATTERN)
  return filenames

### just call this ez func
def get_dataset(bucket_dir, shuffle=False, batch=False):
  return load_dataset(get_tfrec_names(bucket_dir), shuffle, batch)

###### Execution ######
