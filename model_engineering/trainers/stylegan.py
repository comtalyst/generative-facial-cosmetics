########## StyleGan trainer ##########

###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model, optimizers, losses
else:
  from tensorflow.keras import layers, Model, optimizers, losses
from utils.model_utils import *
from utils.generator_utils import *
import functools
import time
import os
from IPython import display

###### Constants ######

FIRSTSTEP = True

## optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

## checkpoints manager
checkpoint_dir = os.path.join(DIR_OUTPUT, os.path.join('training_checkpoints', 'current'))
checkpoint_prefix_name = "ckpt"
checkpoint_prefix = os.path.join(checkpoint_dir, checkpoint_prefix_name)
EPOCHS_TO_SAVE = 1
MAX_TO_KEEP = 2
RESTORE_CHECKPOINT = True

###### Functions ######

## losses
def discriminator_loss(real_output, fake_output):
  real_loss = losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.SUM)(tf.ones_like(real_output), real_output)
  fake_loss = losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.SUM)(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.SUM)(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(generator, discriminator, images, batch_size, strategy):
  latent_size = generator.LATENT_SIZE
  def true_step(images):
    noise = tf.random.normal([batch_size, latent_size])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator.model(noise, training=True)

      real_output = discriminator.model(images, training=True)
      fake_output = discriminator.model(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.model.trainable_variables))
  if isColab():
    strategy.run(true_step, args=(images,))       # make sure this is a tuple using ','
  else:
    true_step(images)

def train(generator, discriminator, dataset, epochs, batch_size, strategy):
  global FIRSTSTEP
  ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                             discriminator_optimizer=discriminator_optimizer,
                             generator=generator.model,
                             discriminator=discriminator.model)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=MAX_TO_KEEP)

  # if a checkpoint exists, restore the latest checkpoint.
  if RESTORE_CHECKPOINT and ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      last_epoch = int(os.path.split(ckpt_manager.latest_checkpoint)[1][len(checkpoint_prefix_name)+1:])
      print ('Latest checkpoint restored: ' + str(last_epoch))
      
  else:
    last_epoch = 0

  allstart = time.time()
  for epoch in range(last_epoch, epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(generator, discriminator, image_batch, batch_size, strategy)
      if FIRSTSTEP:
        print("First batch done!")
        FIRSTSTEP = False

    # Produce images for the GIF as we go
    try:
      display.clear_output(wait=True)
    except:
      pass
    generate_and_save_images(generator, epoch + 1)

    print ('Time for epoch {} is {} sec, total {} sec'.format(epoch + 1, time.time()-start, time.time()-allstart))
    # Save the model every EPOCHS_TO_SAVE epochs (not include in time)
    if (epoch + 1) % EPOCHS_TO_SAVE == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
    

  # saving last epoch, unless it has been saved
  if epochs % EPOCHS_TO_SAVE != 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epochs, ckpt_save_path))
 
