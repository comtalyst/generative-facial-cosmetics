########## Database model ##########

###### Global Imports ######
from sqlalchemy import create_engine  
import json

###### Project Imports ######

###### Constants ######

###### Global Variables ######
db = None

###### Functions ######

def init():
  with open("./keys/keys.json") as jf:
    keys_dict = json.loads(jf.read())
  db_string = keys_dict["db_string"]

  global db
  db = create_engine(db_string, client_encoding='utf8')
  db.execute("CREATE TABLE IF NOT EXISTS api_losses (base_img text, style_img text, base_loss real, style_loss real)")  

def add_row(base_img, style_img, base_loss, style_loss):
  db.execute(f"INSERT INTO api_losses " + 
             f"VALUES (\'{base_img}\', \'{style_img}\', {base_loss}, {style_loss})")

###### Execution ######

init()
