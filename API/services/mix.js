/*
Upper service layer of mixing operation 
Responsible for calling lower services and db models
Exposed to the controller
*/

const stylemixer = require('stylemixer_model.js')

exports.hello = async (text1, text2) => {
  console.log('mix: hello')
  return text1 + 'hey' + text2
}

// call tf service and store in db
exports.mix = async (base_img, style_img) => {
  console.log('Mixing two images...')
  // TODO: revalidate images size + verify inputs
  // TODO: resize image
  mixed_img, loss = stylemixer.mix(base_img, style_img)
  // TODO: contact db
  return mixed_img
}