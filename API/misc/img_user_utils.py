########## Image user utilities ##########
"""
Just for testing and manual tasks
"""

###### Imports ######
import base64

###### Constants ######

###### Global Variables ######

###### Functions ######
def bytes_to_base64str(bbytes):
  # for more info on this algo, see experiment section of finalization
  b64 = base64.b64encode(bbytes)
  b64str = str(b64, "ANSI")
  return b64str

def base64str_to_bytes(b64str):
  b64 = bytes(b64str, "ANSI")
  bbytes = base64.b64decode(b64)
  return bbytes

def img_to_base64str(img_path, output_path=None):
  with open(img_path, "rb") as f:
    img_bytes = f.read()
  if output_path == None:
    output_path = img_path + ".txt"
  b64str = bytes_to_base64str(img_bytes)
  with open(output_path, "w") as f:
    f.write(b64str)

def base64str_to_img(b64str_path, output_path=None):
  with open(b64str_path, "r") as f:
    b64str = f.read()
  if output_path == None:
    output_path = b64str_path + ".png"
  img_bytes = base64str_to_bytes(b64str)
  with open(output_path, "wb") as f:
    f.write(img_bytes)

###### Execution ######
#img_to_base64str("test_2.png")
#img_to_base64str("purple_lips.png")
#base64str_to_img("test_2.png.txt")
#base64str_to_img("purple_lips.png.txt")
base64str_to_img("mixed.txt")