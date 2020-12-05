########## Controller and Interface ##########

###### Global Imports ######
from flask import Flask, request, jsonify
from flask_cors import CORS

###### Project Imports ######
from services import stylemix

###### Constants ######

###### Preprocesses ###### 
app = Flask(__name__)
CORS(app)

###### Global Variables ######

###### Routes ######

# TODO: async stuff
# TODO: initializer

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
  try:
    base_img = str(request.json.get("base_img"))
    style_img = str(request.json.get("style_img"))
    results = stylemix.mix(base_img, style_img)
    print("Controller: mix:: Done!")
    return jsonify(result_img=results)
  except Exception as e:
    print("Controller: mix:: Exception thrown: " + str(e))
    return jsonify(message=str(e)), 400
  

###### Execution ######
if __name__ == '__main__':
  app.run(host="0.0.0.0")