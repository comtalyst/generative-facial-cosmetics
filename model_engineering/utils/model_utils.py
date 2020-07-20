########## Model Utilities ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers
else:
  from tensorflow.keras import layers, Model

###### Constants ######

###### Functions ######

def set_trainable(model, trainable):
  for layer in model.layers:
    layer.trainable = trainable

###### Execution ######
