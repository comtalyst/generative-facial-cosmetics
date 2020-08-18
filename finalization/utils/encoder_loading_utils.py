########## Encoder Utilities ##########

###### Imports ######

from config import *
import os
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from models.encoder import Encoder

###### Constants ######

###### Functions ######

def load_encoder_checkpoint(strategy, model_type=None encoder_ckpt_dir=None):
  if encoder_ckpt_dir == None:
    encoder_ckpt_dir = os.path.join(DIR_OUTPUT, 'encoder_checkpoints')

  encoder = Encoder(strategy, model_type)
  with strategy.scope():
    encoder.model.load_weights(checkpoint_path)
  return encoder

###### Execution ######
