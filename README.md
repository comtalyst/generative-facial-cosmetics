# Generative Facial Cosmetics
Our task is to apply various facial makeup products to the input face automatically and smoothly. Instead of placing the predefined layer of makeup on top of the face image, we are going to approach this task by regenerating the resulting image using Nvidia StyleGAN to apply the makeups smoothly.<br><br>
Currently, only lipstick applying is being worked on. Other features (e.g. eyeliners) are currently not planned. We do not expect the overall process to change entirely, but some minor details are likely to change.<br><br>

<i>API</i>:&nbsp;&nbsp;&nbsp;&nbsp;A RESTful API implemented with Flask for utilizing this model in practice<br>
<i>dataset_engineering</i>:&nbsp;&nbsp;&nbsp;&nbsp;Image dataset analysis and transformation/preprocessing<br>
<i>model_engineering</i>:&nbsp;&nbsp;&nbsp;&nbsp;Core model (GAN) training and optimization<br>
<i>encoder_engineering</i>:&nbsp;&nbsp;&nbsp;&nbsp;Encoder model training and optimization<br><br>

For methodology explanations and implementation details, please visit <a href="https://comtalyst.com/#/pages/GFC">this page</a>.<br><br>

<b>This project is a personal project and is not affiliated with any instituitions or organizations.</b>
