########## Input Data Pipeline ##########

###### Imports ######

from config import *
import os
import tensorflow as tf
from technical.accelerators import BATCH_SIZE as bs
from PIL import Image
import json
import cv2
from models.generator import Generator
import numpy as np

###### Constants ######

DEFAULT_IMAGE_SIZE = (1024, 1024)
BATCH_SIZE = bs
## to be defined
IMAGE_SIZE = None

###### Functions ######

### preprocess input (customized)
def preprocess(image):
  return image

### just call this ez func
def load_image(path, image_size=DEFAULT_IMAGE_SIZE):
  global IMAGE_SIZE
  IMAGE_SIZE = image_size

  img = cv2.imread(path)
  img = preprocess(img)
  return img

###### Execution ######

