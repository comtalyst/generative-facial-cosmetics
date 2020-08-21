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

PAD = 10        
LIPS_UPPER = 340
LIPS_BOX_SIZE = LIPS_UPPER + PAD*2      # must = trained models' preferred size (currently 360)

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

def detect_and_crop_lips(img_path=None, img_bytes=None, img_full=None):
  img = None
  if type(img_full) != type(None):
    img = img_full
  elif type(img_bytes) != type(None):              # bytes provided
    img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)
  elif type(img_path) != type(None):             # path provided
    img = cv2.imread(img_path)
  else:
    raise ValueError('No images provided')

  # preprocess
  
  # get faces and landmarks
  landmarks, face = detect_landmarks(img_full=img, show=False)

  #### filtering faces #####
  if face == None:
    # no face detected
    print('Error: No face detected')
    return None
  
  # get face size details
  face_w = np.abs(face.right() - face.left())
  face_h = np.abs(face.top() - face.bottom())

  if face_w < 400 or face_h < 400:
    # anomaly
    print('Warning: Anomalous input: face size is too small')

  ##### statisticize lips #####
  # get landmarks of lips
  lips_landmarks = landmarks[48:68]     # based on the map far above (currently for 68 landmarks predictor)

  # get lips size details
  lips_landmarks_x = list(landmark[0] for landmark in lips_landmarks)
  lips_landmarks_y = list(landmark[1] for landmark in lips_landmarks)
  min_x = np.min(lips_landmarks_x)
  max_x = np.max(lips_landmarks_x)
  min_y = np.min(lips_landmarks_y)
  max_y = np.max(lips_landmarks_y)
  lips_w = np.abs(max_x - min_x)
  lips_h = np.abs(max_y - min_y)
  lips_w_r = lips_w/face_w
  lips_h_r = lips_h/face_h

  # change from cv2's BGR to RGBA system
  img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
  img = Image.fromarray(img).convert("RGBA")

  # add paddings
  outer_lips_landmarks_x = lips_landmarks_x[0:60-48]          # based on face map
  outer_lips_landmarks_y = lips_landmarks_y[0:60-48]
  center_x = (np.max(outer_lips_landmarks_x) + np.min(outer_lips_landmarks_x)) // 2
  center_y = (np.max(outer_lips_landmarks_y) + np.min(outer_lips_landmarks_y)) // 2
  for i in range(0, 60-48):
    dx = lips_landmarks_x[i] - center_x
    dy = -(lips_landmarks_y[i] - center_y)
    d = (dx**2 + dy**2)**0.5
    try:
      lips_landmarks_x[i] = int(lips_landmarks_x[i] + PAD*(dx/d))
    except:
      print('Divided by zero, skip padding at this point')
    try:
      lips_landmarks_y[i] = int(lips_landmarks_y[i] - PAD*(dy/d))
    except:
      print('Divided by zero, skip padding at this point')
    lips_landmarks[i] = (lips_landmarks_x[i], lips_landmarks_y[i])
  
  # update stats after padding
  min_x = np.min(lips_landmarks_x)
  max_x = np.max(lips_landmarks_x)
  min_y = np.min(lips_landmarks_y)
  max_y = np.max(lips_landmarks_y)
  lips_w = np.abs(max_x - min_x)
  lips_h = np.abs(max_y - min_y)
  lips_w_r = lips_w/face_w
  lips_h_r = lips_h/face_h

  # repeat some cyclehead dots to complete each of two cycles
  lips_landmarks = lips_landmarks[:60-48] + lips_landmarks[0:1] + lips_landmarks[60-48:] + lips_landmarks[60-48:60-48+1]

  # polygon crop (filter)
  cropped_lips = polygon_crop(img, lips_landmarks)

  # undo
  lips_landmarks = lips_landmarks[:60-48] + lips_landmarks[60-48+1:-1]

  # square crop
  cropped_lips = cropped_lips.crop((min_x, min_y, max_x, max_y))
  if lips_w > LIPS_BOX_SIZE:
    size_multiplier = LIPS_BOX_SIZE/lips_w
    cropped_lips = cropped_lips.resize((int(lips_w*size_multiplier), int(lips_h*size_multiplier)))
    lips_w *= size_multiplier
    lips_h *= size_multiplier
  elif lips_h > LIPS_BOX_SIZE:
    size_multiplier = LIPS_BOX_SIZE/lips_h
    cropped_lips = cropped_lips.resize((int(lips_w*size_multiplier), int(lips_h*size_multiplier)))
    lips_w *= size_multiplier
    lips_h *= size_multiplier
  lips_w = int(lips_w)
  lips_h = int(lips_h)

  box = Image.new('RGBA', (LIPS_BOX_SIZE, LIPS_BOX_SIZE), (0, 0, 0, 0))
  offset = ((LIPS_BOX_SIZE - lips_w) // 2, (LIPS_BOX_SIZE - lips_h) // 2)
  box.paste(cropped_lips, offset)

  # update lips landmark
  for i in range(0, len(lips_landmarks)):
    lips_landmarks_x[i] -= min_x - offset[0]
    lips_landmarks_y[i] -= min_y - offset[1]
    lips_landmarks[i] = (int(lips_landmarks_x[i] - (min_x - offset[0])), int(lips_landmarks_y[i] - (min_y - offset[1])))
  
  return box, [min_x, min_y, max_x, max_y, offset[0], offset[1]]

def replace_lips(cropped_lips, p_data, img_path=None, img_bytes=None, img_full=None):
  img = None
  if type(img_full) != type(None):
    img = img_full
  elif type(img_bytes) != type(None):              # bytes provided
    img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)
  elif type(img_path) != type(None):             # path provided
    img = cv2.imread(img_path)
  else:
    raise ValueError('No images provided')

  min_x, min_y, max_x, max_y, offset_x, offset_y = p_data
  cropped_lips_pil = Image.fromarray((cropped_lips*255).astype('uint8'))
  cropped_lips_pil = cropped_lips_pil.crop((offset_x, offset_y, offset_x+(max_x-min_x), offset_y+(max_y-min_y)))
  img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
  img_pil.paste(cropped_lips_pil, (min_x, min_y), cropped_lips_pil)
  plt.imshow(img_pil)

  return img_pil


###### Execution ######
