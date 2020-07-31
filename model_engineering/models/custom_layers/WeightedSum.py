########## WeightedSum Layer ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, backend
else:
  from tensorflow.keras import layers, backend

###### Class Content ######
class WeightedSum(layers.Add):

  ###### Constants ######

	###### Constructor ######

  def __init__(self, alpha=0.0, **kwargs):
    super(WeightedSum, self).__init__(**kwargs)
    #self.alpha = backend.variable(alpha, name='ws_alpha')
    self.alpha = tf.Variable(alpha, name='ws_alpha', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

  ###### Overrides ######
  def _merge_function(self, inputs):
    # only supports a weighted sum of two inputs
    assert (len(inputs) == 2)
    # ((1-a) * input1) + (a * input2)
    output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
    return output

