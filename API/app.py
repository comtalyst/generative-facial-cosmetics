########## Controller ##########

###### Imports ######
from flask import Flask, request, jsontify

###### Constants ######

########## Preprocesses ########## 

app = Flask(__name__)

########## Global Variables ##########

###### Routes ######

@app.route('/hello', methods=['POST'])
def hello():
  text1 = request.form.get("text1")
  text2 = request.form.get("text2")
  return jsontify(f"{text1} {hello} {text2}")

###### Execution ######
