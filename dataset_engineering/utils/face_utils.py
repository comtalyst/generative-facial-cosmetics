########## Landmark Detection and Cropping Utilities ##########

###### Imports ######

from config import *
import cv2
import numpy as np
import os
import dlib
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt 

if isColab():
  from google.colab.patches import cv2_imshow
else:
  from cv2 import imshow as cv2_imshow

###### Constants ######

POINTS = 68

###### Functions ######

### detect landmarks and show the results
def detect_landmarks(img_path=None, img_bytes=None, img_full=None, show=False):
  # Load the detector
  detector = dlib.get_frontal_face_detector()

  # Load the predictor
  predictor = dlib.shape_predictor(os.path.join(DIR, os.path.join("utils", "shape_predictor_68_face_landmarks.dat")))

  # read the image
  img = None
  if type(img_full) != type(None):
    img = img_full
  elif type(img_bytes) != type(None):              # bytes provided
    img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)
  elif type(img_path) != type(None):             # path provided
    img = cv2.imread(img_path)
  else:
    raise ValueError('No images provided')

  # Convert image into grayscale
  gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

  # Use detector to find landmarks
  faces = detector(gray)

  # if no face detected
  if faces == None or len(faces) == 0:
    return None, None

  max_size = 0
  # if there are multiple faces, select the largest
  for face_i in faces:
    x1 = face_i.left() # left point
    y1 = face_i.top() # top point
    x2 = face_i.right() # right point
    y2 = face_i.bottom() # bottom point
    if np.abs(x2-x1)*np.abs(y2-y1) > max_size:
      max_size = np.abs(x2-x1)*np.abs(y2-y1)
      face = face_i

  x1 = face.left() # left point
  y1 = face.top() # top point
  x2 = face.right() # right point
  y2 = face.bottom() # bottom point

  # Create landmark object
  landmarks = predictor(image=gray, box=face)
  img_dots = img.copy()

  # Loop through all the points
  for n in range(0, POINTS):
    x = landmarks.part(n).x
    y = landmarks.part(n).y

    # Draw a circle
    cv2.circle(img=img_dots, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

  # show the image
  if show:
    cv2_imshow(cv2.resize(img_dots, (512, 512)) )

  # Delay between every fram
  cv2.waitKey(delay=0)

  # Close all windows
  cv2.destroyAllWindows()

  landmarks_list = list((landmarks.part(n).x, landmarks.part(n).y) for n in range(0, POINTS))
  return landmarks_list, face

def polygon_crop(img, polygon):
  # convert to numpy (for convenience)
  imArray = np.asarray(img)

  # create mask
  maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
  ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
  mask = np.array(maskIm)

  # assemble new image (uint8: 0-255)
  newImArray = np.empty(imArray.shape,dtype='uint8')

  # colors (three first columns, RGB)
  newImArray[:,:,:3] = imArray[:,:,:3]

  # transparency (4th column)
  newImArray[:,:,3] = mask*255

  # back to Image from numpy
  newIm = Image.fromarray(newImArray, "RGBA")

  return newIm

def detect_and_crop_lips(img_path=None, img_bytes=None, img_full=None, show=False):
  # read the image
  img = None
  if type(img_full) != type(None):
    img = img_full
  elif type(img_bytes) != type(None):              # bytes provided
    img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)
  elif type(img_path) != type(None):             # path provided
    img = cv2.imread(img_path)
  else:
    raise ValueError('No images provided')

  # get faces and landmarks
  landmarks, _ = detect_landmarks(img_full=img, show=False)

  # change from cv2's BGR to RGB system
  img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

  # get landmarks of lips
  lips_landmarks = landmarks[48:68]     # based on the map far above (currently for 68 landmarks predictor)

  # read image as RGB and add alpha (transparency)
  img = Image.fromarray(img).convert("RGBA")

  # repeat some cyclehead dots to complete each of two cycles
  lips_landmarks = lips_landmarks[:60-48] + lips_landmarks[0:1] + lips_landmarks[60-48:] + lips_landmarks[60-48:60-48+1] 

  # polygon crop
  cropped_lips = polygon_crop(img, lips_landmarks)

  if show:
    plt.imshow(cropped_lips)

  return cropped_lips

###### Execution ######
