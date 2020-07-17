########## Model Utilities ##########

###### Imports ######

import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.api._v2.keras import layers

###### Constants ######

###### Functions ######

def set_trainable(model, trainable):
  for layer in model.layers:
    layer.trainable = trainable

###### Execution ######