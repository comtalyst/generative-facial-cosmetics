########## StyleGan trainer ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model, optimizers, losses
else:
  from tensorflow.keras import layers, Model, optimizers, losses
from utils.model_utils import *
from utils.generator_utils import *
import functools
import time
import os
from IPython import display

###### Constants ######

FIRSTSTEP = True

## optimizers
DEFAULT_LR = 1e-3
optimizer_type = tf.keras.optimizers.Adam

## checkpoints manager
checkpoint_dir = os.path.join(DIR_OUTPUT, os.path.join('training_checkpoints', 'current'))
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')
#EPOCHS_TO_SAVE = 1
#MAX_TO_KEEP = 100

###### Functions ######

# load checkpoint without training, intended to have similar function as loading models
def load_checkpoint(model, strategy):
  with strategy.scope():
    model.load_weights(checkpoint_path)

def train(encoder, training_dataset, validation_dataset, epochs, strategy, lr=DEFAULT_LR, restore_checkpoint=True):
  model = encoder.model
  if restore_checkpoint:
    load_checkpoint(model, strategy)
  with strategy.scope():
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                       save_weights_only=True,
                                                       verbose=1)
    model.compile(optimizer = optimizer_type(lr), loss = 'mse', metrics = ['mse'])
    # dataset is batched, no need to re-batch
    model.fit(training_dataset, epochs = epochs, validation_data=validation_dataset, callbacks=[ckpt_callback])
 
