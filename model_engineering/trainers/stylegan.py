########## StyleGan trainer ##########
'''
Next time do this as a class for simplicity of prop customizations
'''
###### Imports ######

from config import *
import tensorflow as tf
from tensorflow import keras
if isWindows():
  from tensorflow_core.python.keras.api._v2.keras import layers, Model, optimizers, losses, backend
else:
  from tensorflow.keras import layers, Model, optimizers, losses, backend
from utils.model_utils import *
from utils.generator_utils import *
import functools
import time
import os
from IPython import display

###### Constants ######

FIRSTSTEP = True
DEFAULT_LR = 1e-4
gen_loss_dict = dict()
disc_loss_dict = dict()

## optimizers
generator_optimizer = tf.keras.optimizers.Adam(DEFAULT_LR)
discriminator_optimizer = tf.keras.optimizers.Adam(DEFAULT_LR)

## checkpoints manager
checkpoint_dir = os.path.join(DIR_OUTPUT, os.path.join('training_checkpoints', 'current'))
EPOCHS_TO_SAVE = 1
MAX_TO_KEEP = 100

###### Functions ######

### losses
def discriminator_loss(real_output, fake_output):
  ## minimax loss
  real_loss = losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.SUM)(tf.ones_like(real_output), real_output)
  fake_loss = losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.SUM)(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss
  ## wasserstein loss: the higher discriminator (critic) output, the more "real" the image is
  ###### ALSO CLIP WEIGHTS WHEN USING WASSERSTEIN
  #return -(backend.mean(real_output) - backend.mean(fake_output))

def generator_loss(fake_output):
  ## minimax loss
  return losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.SUM)(tf.ones_like(fake_output), fake_output)
  ## wasserstein loss
  #return -backend.mean(fake_output)

@tf.function
def train_step(generator, discriminator, epoch, fade_epochs, images, batch_size, strategy):
  latent_size = generator.LATENT_SIZE
  gen_loss_dict[epoch] = list()
  disc_loss_dict[epoch] = list()
  def true_step(images):
    noise = tf.random.normal([batch_size, latent_size])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      if epoch >= fade_epochs or generator.model_fade == None:
        generated_images = generator.model(noise, training=True)
        real_output = discriminator.model(images, training=True)
        fake_output = discriminator.model(generated_images, training=True)
      else:
        generator.setAlpha(epoch/fade_epochs)
        discriminator.setAlpha(epoch/fade_epochs)
        generated_images = generator.model_fade(noise, training=True)
        real_output = discriminator.model_fade(images, training=True)
        fake_output = discriminator.model_fade(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
      gen_loss_dict[epoch].append(gen_loss)
      disc_loss_dict[epoch].append(disc_loss)
    return gen_loss, disc_loss

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.model.trainable_variables))
  if isColab():
    gen_loss, disc_loss = strategy.run(true_step, args=(images,))       # make sure this is a tuple using ','
  else:
    gen_loss, disc_loss = true_step(images)
  return strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)

# load checkpoint without training, intended to have similar function as loading models
# do not load fade if you are loading from models that do not have fade
# also, loading fade is not neccesary if you are going to progress it
def load_checkpoint(generator, discriminator, strategy, load_fade=True):
  with strategy.scope():
    if load_fade:
      ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator.model,
                                discriminator=discriminator.model,
                                generator_fade=generator.model_fade,
                                discriminator_fade=discriminator.model_fade)
    else:
      ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator.model,
                                discriminator=discriminator.model)
    checkpoint_dir_progress = os.path.join(checkpoint_dir, str(generator.current_progress))
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir_progress, max_to_keep=MAX_TO_KEEP)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        last_epoch = int(os.path.split(ckpt_manager.latest_checkpoint)[1][5:])
        print ('Latest checkpoint restored: ' + str(last_epoch))
    else :
      print("No checkpoints to be restored")

def train(generator, discriminator, dataset, fade_epochs, epochs, batch_size, strategy, lr=[DEFAULT_LR, DEFAULT_LR], restore_checkpoint=True):
  global generator_optimizer
  global discriminator_optimizer
  generator_optimizer = tf.keras.optimizers.Adam(lr[0])
  discriminator_optimizer = tf.keras.optimizers.Adam(lr[1])
  #generator_optimizer = tf.keras.optimizers.RMSprop(lr[0])
  #discriminator_optimizer = tf.keras.optimizers.RMSprop(lr[1])
  
  if generator.current_progress != discriminator.current_progress:
    raise ValueError("The progresses of generator " + str(generator.current_progress) + 
                    " and discriminator " + str(discriminator.current_progress) + " are not equal")

  global FIRSTSTEP
  with strategy.scope():
    ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                              discriminator_optimizer=discriminator_optimizer,
                              generator=generator.model,
                              discriminator=discriminator.model,
                              generator_fade=generator.model_fade,          # fade model must exist in order to load
                              discriminator_fade=discriminator.model_fade)  # should not worry since we usually dont save first gen model, but you cannot continue training from old ver.
    checkpoint_dir_progress = os.path.join(checkpoint_dir, str(generator.current_progress))
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir_progress, max_to_keep=MAX_TO_KEEP)

    # if a checkpoint exists, restore the latest checkpoint.
    if restore_checkpoint and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        last_epoch = int(os.path.split(ckpt_manager.latest_checkpoint)[1][5:])
        print ('Latest checkpoint restored: ' + str(last_epoch))
    else :
      if restore_checkpoint:
        print("No checkpoints to be restored")
      elif ckpt_manager.latest_checkpoint:
        print("Checkpoints found, but will be ignored and replaced")
      last_epoch = 0

  allstart = time.time()
  for epoch in range(last_epoch+1, epochs+1):
    start = time.time()
    gen_losses = 0.0
    disc_losses = 0.0
    n_batch = 0

    for image_batch in dataset:
      gen_loss, disc_loss = train_step(generator, discriminator, epoch, fade_epochs, image_batch, batch_size, strategy)
      gen_losses = tf.math.add(gen_losses, gen_loss)
      disc_losses = tf.math.add(disc_losses, disc_loss)
      if FIRSTSTEP:
        print("First batch done!")
        FIRSTSTEP = False
      n_batch += 1
    
    gen_loss = tf.math.divide(tf.math.reduce_sum(gen_losses), n_batch*batch_size)     # may not accurate??
    disc_loss = tf.math.divide(tf.math.reduce_sum(disc_losses), n_batch*batch_size)     # may not accurate??

    # Produce images for the GIF as we go
    '''
    try:
      display.clear_output(wait=True)
    except:
      pass
    '''
    generate_and_save_images(generator, epoch, bool(epoch < fade_epochs))

    print('Epoch {}: Average generator loss = {}, discriminator loss = {}'.format(epoch, gen_loss, disc_loss))
    print ('Time for epoch {} is {} sec, total {} sec'.format(epoch, time.time()-start, time.time()-allstart))
    # Save the model every EPOCHS_TO_SAVE epochs (not include in time)
    if (epoch) % EPOCHS_TO_SAVE == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
    

  # saving last epoch, unless it has been saved
  if epochs % EPOCHS_TO_SAVE != 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epochs, ckpt_save_path))
 
