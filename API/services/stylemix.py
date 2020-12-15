########## Style mix service ##########
"""
Upper service layer of mixing operation 
Responsible for calling lower services and db models
Exposed to the controller
"""

###### Global Imports ######

###### Project Imports ######
from services.backend import ml_model_mix
from services.backend import sizing
from models import db

###### Constants ######

###### Global Variables ######

###### Routes ######

def mix(base_img, style_img):
  print("services.stylemix: mix")
  ## resize here to save db storage space (there is still upload size limit on front-end)
  base_img = sizing.validate_and_resize(base_img)
  style_img = sizing.validate_and_resize(style_img)

  mixed_img, base_loss, style_loss = ml_model_mix.mix(base_img, style_img, return_losses=True)
  db.add_row(base_img, style_img, base_loss, style_loss)
  return mixed_img

###### Execution ######