########## Generator Utilities ##########

###### Imports ######

from config import *
import os
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from models.generator import Generator

###### Constants ######

###### Functions ######

def load_generator_checkpoint(strategy, generator_ckpt_dir=None):
  if generator_ckpt_dir == None:
    generator_ckpt_dir = os.path.join(DIR_OUTPUT, 'generator_checkpoints')

  generator = Generator(strategy)
  while generator.current_progress < 5:
    generator.progress(strategy)
  ckpt = tf.train.Checkpoint(generator=generator.model)
  ckpt_manager = tf.train.CheckpointManager(ckpt, generator_ckpt_dir, max_to_keep=1)
  ckpt.restore(ckpt_manager.latest_checkpoint)
  return generator

###### Execution ######
