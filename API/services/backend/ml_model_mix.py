########## TF interface for mixing ##########
"""
Lower service layer of mixing operation
Responsible for interacting with tf library
Exposed to upper service (e.g. mix.py)
"""

###### Global Imports ######
import tensorflow as tf

###### Project Imports ######

###### Constants ######

###### Global Variables ######

###### Routes ######

def init():
  pass
  # TODO: init

def mix(base_img, style_img):
  print("services.backend.ml_model_mix: mix")
  # TODO: return mixed image

###### Execution ######