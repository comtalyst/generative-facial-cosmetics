########## Input Data Pipeline ##########

###### Imports ######

from config import *
import os
import tensorflow as tf
from technical.accelerators import BATCH_SIZE as bs
from PIL import Image
import json

###### Constants ######

AUTO = tf.data.experimental.AUTOTUNE
DEFAULT_IMAGE_SIZE = [360, 360]
BATCH_SIZE = bs
JSON = None

###### Functions ######

### convert tfrecord of PNGs to image tensors
def read_tfrecord(example):
  features = {
    "image": tf.io.FixedLenFeature([], tf.string),        # tf.string means bytestring
    "id": tf.io.FixedLenFeature([], tf.int64)
  }
  example = tf.io.parse_example(example, features)
  images = tf.image.decode_png(example['image'], channels=4)
  images = tf.cast(images, tf.float32) / 255.0              # convert image to floats in [0, 1] range
  images = tf.image.resize(images, IMAGE_SIZE)              # explicit size will be needed for TPU
  ids = example['id']
  return (images, ids)

### data augmentation (return the image param if no augmentation is needed)
def data_augment(image, id):
  ## random resize (zoom)
  if str(id)[0] != 'T':
    coords = JSON[str(id)]
    min_x = 9999999
    min_y = 9999999
    max_x = -9999999
    max_y = -9999999
    for coord in coords:
      x, y = coord
      min_x = np.min(min_x, x)
      min_y = np.min(min_y, y)
      max_x = np.max(max_x, x)
      max_y = np.max(max_y, y)
    w = max_x - min_x
    h = max_y - min_y
    max_resize = np.max(w/IMAGE_SIZE[0], h/IMAGE_SIZE[1])
    random_scale = max_resize + np.random.rand()*(1 - max_resize)   # random resize scale without overflowing the image content
    image = tf.image.central_crop(image, random_scale)              # crop from center
    image = Image.fromarray(image.numpy()).resize(IMAGE_SIZE)       # resize back using PIL
    image = tf.convert_to_tensor(np.array(image))                   # convert back to tf tensor
  
  ## random rotate
  max_degree = 40
  if str(id)[0] != 'T':
    image = tf.convert_to_tensor(tf.keras.preprocessing.image.random_rotation(image.numpy(), max_degree, row_axis=0, col_axis=1, channel_axis=2))

  return (image, id)

### unpack (image, id) to only image
def unpack(image, id):
  return image

### return "list" of image tensors from specified TFRecords
def load_dataset(filenames, shuffle=False, batch=False, augment=False, json_map=None):
  # read from TFRecords. For optimal performance, read from multiple
  # TFRecord files at once and set the option experimental_deterministic = False
  # to allow order-altering optimizations.
  option_no_order = tf.data.Options()
  option_no_order.experimental_deterministic = False

  dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
  dataset = dataset.with_options(option_no_order)
  dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)   # dataset from tfrecords is now in a parallel (normal/good) format
  if augment:
    global JSON 
    JSON = json_map
    dataset = dataset.repeat(2)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
  if shuffle:
    dataset = dataset.shuffle(len(filenames))
  if batch:
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) 
  dataset = dataset.map(unpack, num_parallel_calls=AUTO)
  return dataset

def load_jsons(filenames):
  json_dict = dict()
  for fname in filenames:
    fjson = json.load(open(fname, 'r', encoding='utf-8'))
    json_dict.update(fjson)
  return json_dict

### get filenames
def get_tfrec_names(dir):
  PATH_PATTERN = os.path.join(dir, '*.tfrec')
  filenames = tf.io.gfile.glob(PATH_PATTERN)
  return filenames

def get_json_names(dir):
  PATH_PATTERN = os.path.join(dir, '*.json')
  filenames = tf.io.gfile.glob(PATH_PATTERN)
  return filenames

### just call this ez func
def get_dataset(bucket_dir, shuffle=False, batch=False, augment=False, json_dir=None, image_size=DEFAULT_IMAGE_SIZE):
  global IMAGE_SIZE
  IMAGE_SIZE = image_size
  if augment:
    json_map = load_jsons(get_json_names(json_dir))       # should fit into memory
  else:
    json_map = None
  return load_dataset(get_tfrec_names(bucket_dir), shuffle, batch, augment, json_map)

###### Execution ######
       