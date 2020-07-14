########## TFRecords Manager ##########

###### Imports ######

from config import *
import os
import tensorflow as tf

###### Constants ######

# technical constants
AUTO = tf.data.experimental.AUTOTUNE           # use in dataset management

# architecture/model constants (some of them may be near the respective code)
ORIG_IMAGE_SIZE = [1024, 1024]
IMAGE_SIZE = [1024, 1024]

###### SETUPS ######

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()     # TPU detection
except ValueError:
  tpu = None
  gpus = tf.config.experimental.list_logical_devices("GPU")
  
HARDWARE = 'CPU'
# Select appropriate distribution strategy for hardware
if tpu:
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.experimental.TPUStrategy(tpu)
  print('Running on TPU ', tpu.master())  
  HARDWARE = 'TPU'
elif len(gpus) > 0:
  strategy = tf.distribute.MirroredStrategy(gpus) # this works for 1 to multiple GPUs
  print('Running on ', len(gpus), ' GPU(s) ')
  HARDWARE = 'GPU'
else:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on CPU')

print("Number of accelerators (cores): ", strategy.num_replicas_in_sync)

###### Functions ######

### convert tfrecord of PNGs to image tensors
def read_tfrecord(example):
  with strategy.scope(): 
    features = {
      "image": tf.io.FixedLenFeature([], tf.string),        # tf.string means bytestring
    }
    example = tf.io.parse_example(example, features)
    images = tf.image.decode_png(example['image'], channels=3)
    images = tf.cast(images, tf.float32) / 255.0              # convert image to floats in [0, 1] range
    images = tf.image.resize(images, IMAGE_SIZE)              # explicit size will be needed for TPU
    return images ######
  
### convert tfrecord of PNGs to PNGs
def read_tfrecord_raw(example):
  with strategy.scope(): 
    features = {
      "image": tf.io.FixedLenFeature([], tf.string),        # tf.string means bytestring
    }
    example = tf.io.parse_example(example, features)
    images = example['image']
    return images ######
  
### return "list" of image tensors from specified TFRecords
def load_dataset(filenames, raw=False, shuffle=False):
  # read from TFRecords. For optimal performance, read from multiple
  # TFRecord files at once and set the option experimental_deterministic = False
  # to allow order-altering optimizations.

  option_no_order = tf.data.Options()
  option_no_order.experimental_deterministic = False

  dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
  dataset = dataset.with_options(option_no_order)
  if not raw:
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)   # dataset from tfrecords is now in a parallel (normal/good) format
  else:
    dataset = dataset.map(read_tfrecord_raw, num_parallel_calls=AUTO)   # dataset from tfrecords is now in a parallel (normal/good) format
  if shuffle:
    dataset = dataset.shuffle(len(filenames))
  return dataset

### get filenames
def get_tfrec_names(dir):
  PATH_PATTERN = os.path.join(dir, '*.tfrec')
  filenames = tf.io.gfile.glob(PATH_PATTERN)
  return filenames

### writes new dataset
def save_tfrecord(path = 'tfrecord.tfrec', data=None):
  out_file = tf.io.TFRecordWriter(path)
  for subdata in data:
    feature = {
      "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[subdata['image']])),
      "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[subdata['id']]))
    }
    tf_record = tf.train.Example(features=tf.train.Features(feature=feature))
    out_file.write(tf_record.SerializeToString())
  out_file.close()

###### Execution ######
