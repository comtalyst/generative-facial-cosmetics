########## Image utilities ##########

###### Global Imports ######
import base64

###### Project Imports ######

###### Constants ######

###### Global Variables ######

###### Routes ######

def bytes_to_base64str(bbytes):
  # for more info on this algo, see experiment section of finalization
  b64 = base64.b64encode(bbytes)
  b64str = str(b64, "utf-8")
  return b64str

def base64str_to_bytes(b64str):
  b64 = bytes(b64str, "utf-8")
  bbytes = base64.b64decode(b64)
  return bbytes

###### Execution ######