/*
Lower service layer of mixing operation
Responsible for interacting with tf library
Exposed to upper service (e.g. mix.js)
*/

const tf = require('@tensorflow/tfjs-node')

// inits
const generator = await tf.loadLayersModel('../resources/generator/model.json')
const encoder = await tf.loadLayersModel('../resources/encoder/model.json')


exports.hello = async (text1, text2) => {
  console.log('stylemixer_model: hello')
  return text1 + 'hey' + text2
}

// use tf to mix style
exports.mix = async (base_img, style_img) => {
  console.log('Using tensorflow...')
  // TODO: tf operations
  return mixed_img
}