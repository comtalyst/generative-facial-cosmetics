########## Image size management ##########
"""
Exposed to upper service (e.g. mix.py)
"""

###### Global Imports ######
import cv2
import numpy as np
import base64

###### Project Imports ######

###### Constants ######
preferred_size = (360, 360)

###### Global Variables ######

###### Routes ######

def bytes_to_base64str(bbytes):
  # for more info on this algo, see experiment section of finalization
  b64 = base64.b64encode(bbytes)
  b64str = str(b64, "ANSI")
  return b64str

def base64str_to_bytes(b64str):
  b64 = bytes(b64str, "ANSI")
  bbytes = base64.b64decode(b64)
  return bbytes

def validate_and_resize(img_b64str):
  print("services.backend.sizing: validate_and_resize")
  img_bytes = base64str_to_bytes(img_b64str)
  img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)

  if img.shape[1] != img.shape[0]:
    raise RuntimeError("INVALID_IMAGE_SIZE")
  img = cv2.resize(img, preferred_size)

  img_new_bytes = cv2.imencode(".png", img)[1].tostring()
  img_new_b64str = bytes_to_base64str(img_new_bytes)
  return img_new_b64str

###### Execution ######