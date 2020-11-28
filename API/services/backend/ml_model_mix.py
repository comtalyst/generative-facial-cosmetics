########## TF interface for mixing ##########
"""
Lower service layer of mixing operation
Responsible for interacting with tf library
Exposed to upper service (e.g. mix.py)
"""

###### Global Imports ######
import tensorflow as tf
import numpy as np
import cv2

from services.backend.utils.img_utils import base64str_to_bytes, bytes_to_base64str
from services.backend.utils.face_utils import detect_and_crop_lips, replace_lips

###### Project Imports ######

###### Constants ######
generator_path = "./resources/generator.h5"
encoder_path = "./resources/encoder.h5"

###### Global Variables ######
generator = None
encoder = None

###### Functions ######

def init():
  print("services.backend.ml_model_mix: init")
  global generator
  global encoder

  ## custom layer to be provided when loading model
  def AdaIN(x):
      mean = tf.keras.backend.mean(x[0], axis = [1, 2], keepdims = True)
      std = tf.keras.backend.std(x[0], axis = [1, 2], keepdims = True) + 1e-7
      y = (x[0] - mean) / std
      pool_shape = [-1, 1, 1, y.shape[-1]]
      scale = tf.keras.backend.reshape(x[1], pool_shape)
      bias = tf.keras.backend.reshape(x[2], pool_shape)
      return y * scale + bias

  generator = tf.keras.models.load_model(generator_path, custom_objects={'AdaIN': AdaIN})
  encoder = tf.keras.models.load_model(encoder_path)

def image_mse(img1, img2):
  return float(tf.keras.losses.MSE(tf.keras.backend.flatten(img1), tf.keras.backend.flatten(img2)))



def mix(base_img_b64str, style_img_b64str, return_losses=False):
  print("services.backend.ml_model_mix: mix")

  ## read in images
  base_img_bytes = base64str_to_bytes(base_img_b64str)
  base_img = cv2.imdecode(np.fromstring(base_img_bytes, np.uint8), cv2.IMREAD_COLOR)
  style_img_bytes = base64str_to_bytes(style_img_b64str)
  style_img = cv2.imdecode(np.fromstring(style_img_bytes, np.uint8), cv2.IMREAD_COLOR)

  ## crop
  base_cropped_lips, p_data, base_img = detect_and_crop_lips(img_full=base_img)
  base_cropped_lips = np.array(base_cropped_lips)/255
  style_cropped_lips, _, style_img = detect_and_crop_lips(img_full=style_img)
  style_cropped_lips = np.array(style_cropped_lips)/255

  ## encode
  base_encoded = encoder(np.expand_dims(base_cropped_lips, 0))
  style_encoded = encoder(np.expand_dims(style_cropped_lips, 0))

  ## regenerate
  mixed_regenerated_lips = generator([base_encoded]*6 + [style_encoded*2], training=False)[0]

  ## replace
  mixed_img = replace_lips(mixed_regenerated_lips.numpy(), p_data, img_full=base_img)

  ## convert
  mixed_img_bgr = cv2.cvtColor(np.array(mixed_img), cv2.COLOR_RGB2BGR)
  mixed_img_bytes = cv2.imencode(".png", mixed_img_bgr)[1].tostring()
  mixed_img_b64str = bytes_to_base64str(mixed_img_bytes)

  ## find loss of both images: MSE between original and regenerated (uninjected) lips
  if return_losses:
    # note that it is not true encoded loss since the last layer is penalized
    ## regenerate
    base_regenerated_lips = generator([base_encoded]*6 + [base_encoded*2], training=False)[0]
    style_regenerated_lips = generator([style_encoded]*6 + [style_encoded*2], training=False)[0]
    ## calculate
    base_regeneration_loss = image_mse(base_cropped_lips, base_regenerated_lips)
    style_regeneration_loss = image_mse(style_cropped_lips, style_regenerated_lips)

    return mixed_img_b64str, base_regeneration_loss, style_regeneration_loss

  return mixed_img_b64str

###### Execution ######
init()