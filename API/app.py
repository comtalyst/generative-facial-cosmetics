########## Controller and Interface ##########

###### Global Imports ######
from flask import Flask, request, jsonify

###### Project Imports ######
from services import stylemix

###### Constants ######

###### Preprocesses ###### 
app = Flask(__name__)

###### Global Variables ######

###### Routes ######

@app.route('/hello', methods=['POST'])
def hello():
  print("Controller: hello")
  text1 = request.form.get("text1")
  text2 = request.form.get("text2")
  return jsonify(f"{text1} hello {text2}")

@app.route('/mix', methods=['POST'])
def mix():
  """
  Expecting two square images (the images will be cropped manually on the front-end)
  encoded as a ANSI string rep. of base64(image bytes)
  """
  print("Controller: mix")
  base_img = str(request.form.get("base_img"))
  style_img = str(request.form.get("style_img"))
  results = stylemix.mix(base_img, style_img)
  return jsonify(results)

###### Execution ######
if __name__ == '__main__':
  app.run(host="0.0.0.0")