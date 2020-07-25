########## Generator Utilities ##########

###### Imports ######

from config import *
import os
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

###### Constants ######

###### Functions ######

def generate_and_save_images(generator, epoch):
  latent_size = generator.LATENT_SIZE
  test_input = tf.random.normal([16, latent_size])

  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = generator.model(test_input, training=False)

  try:
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig(os.path.join(os.path.join(DIR_OUTPUT, 'training_generateds'), 'image-progress_{:02d}-epoch_{:04d}.png'.format(generator.current_progress, epoch)))
    plt.show()
  except:
    print('Figure showing failed')

###### Execution ######
