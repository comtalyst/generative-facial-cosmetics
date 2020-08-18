########## Accelerator Manager ##########
'''
just export the variable 'strategy'
'''

###### Imports ######

from config import *
import os
import tensorflow as tf

###### Constants ######

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

if tpu:
  BATCH_SIZE = int(16*strategy.num_replicas_in_sync)  # A TPU has 8 cores so this will be 128
else:
  BATCH_SIZE = 32  # On Colab/GPU, a higher batch size does not help and sometimes does not fit on the GPU (OOM)

###### Functions ######

###### Execution ######
