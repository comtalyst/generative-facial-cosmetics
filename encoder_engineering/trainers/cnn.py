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
import numpy as np

###### Constants ######

FIRSTSTEP = True

## optimizers
DEFAULT_LR = 2e-4
optimizer_type = tf.keras.optimizers.Adam

## checkpoints manager
checkpoint_dir = os.path.join(DIR_OUTPUT, os.path.join('training_checkpoints', 'current'))
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')
#EPOCHS_TO_SAVE = 1
#MAX_TO_KEEP = 100

###### Functions ######

### load checkpoint without training, intended to have similar function as loading models
def load_checkpoint(model, strategy):
  with strategy.scope():
    model.load_weights(checkpoint_path)

### for checkpoints saved using ckpt.save(), model=model
def load_checkpoint_ckpt(model, strategy, path=None):
  if path == None:
    path = tf.train.latest_checkpoint(checkpoint_dir)     # depends on what noted in the "checkpoint." file
    print("latest_checkpoint: " + path)
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.restore(path)

@tf.function
def compute_loss(reals, predictions, loss_weights=None, generator=None):
  # return size = batch size (1 loss per batch)
  # single MSE (default)
  if loss_weights == None:
    return losses.MSE(reals, predictions)

  # mixed losses
  total_loss = 0.0
  for loss_type, loss_w in loss_weights.items():
    # generated image loss
    if loss_type == 'gen':
      real_generateds = generator.model(reals, training=False)
      prediction_generateds = generator.model(predictions, training=False)

      # flatten them to properly find the mean as a value
      batch_size = reals.shape[0]
      generated_size_flat = tf.math.reduce_prod(generator.model.output.shape[1:])     # skip 'none' in the first pos
      new_size = (batch_size, generated_size_flat)
      total_loss = tf.math.add(total_loss, loss_w*losses.MSE(tf.keras.backend.reshape(real_generateds, new_size), tf.keras.backend.reshape(prediction_generateds, new_size)))
    
    # mean squared error
    elif loss_type == 'mse':
      total_loss = tf.math.add(total_loss, loss_w*losses.MSE(reals, predictions))

    # unknown loss
    else:
      raise ValueError('Unknown loss_weights: ' + str(loss_weights))
  return total_loss  

@tf.function
def test_step(model, data_batch, strategy, loss_weights=None, generator=None):
  ## unpack data batch
  images = data_batch[0]
  latents = data_batch[1]

  ## define tf step
  def true_step(images, latents):
    ## forward prop
    logits = model(images, training=False)
    batch_loss = compute_loss(latents, logits, loss_weights, generator)
    return batch_loss

  ## execute the step
  if isColab():
    batch_loss = strategy.run(true_step, args=(images, latents,))       # make sure this is a tuple using ','
  else:
    batch_loss = true_step(images, latents)
  ## return loss
  return strategy.reduce(tf.distribute.ReduceOp.SUM, batch_loss, axis=None)

@tf.function
def train_step(model, data_batch, optimizer, strategy, loss_weights=None, generator=None):
  ## unpack data batch
  images = data_batch[0]
  latents = data_batch[1]

  ## define tf step
  def true_step(images, latents):
    ## forward prop
    with tf.GradientTape() as tape:
      logits = model(images, training=True)
      batch_loss = compute_loss(latents, logits, loss_weights, generator)

    ## apply gradients
    grads = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return batch_loss

  ## execute the step
  if isColab():
    batch_loss = strategy.run(true_step, args=(images, latents,))       # make sure this is a tuple using ','
  else:
    batch_loss = true_step(images, latents)
  ## return loss
  return strategy.reduce(tf.distribute.ReduceOp.SUM, batch_loss, axis=None)

def train(encoder, generator, dataset_gen_func, n_train, n_valid, epochs, strategy, 
          loss_weights=None, lr=DEFAULT_LR, restore_checkpoint=True):
  ## inits
  model = encoder.model
  model_type = encoder.model_type
  global FIRSTSTEP

  ## restore checkpoints (new ver only)
  if restore_checkpoint:
    load_checkpoint(model, strategy)

  ## define optimizer
  optimizer = optimizer_type(lr)

  ## train each epoch
  allstart = time.time()
  for epoch in range(1, epochs+1):
    ## init
    start = time.time()
    training_losses = 0.0
    validation_losses = 0.0
    n_batch_train = 0
    n_batch_valid = 0

    ## fetch datasets
    training_dataset, validation_dataset = dataset_gen_func(generator, strategy, batch=True, 
                                                            n_train=n_train, n_valid=n_valid, model_type=model_type)
    
    ## training steps
    for data_batch in training_dataset:
      training_losses = tf.math.add(training_losses, train_step(model, data_batch, optimizer, strategy, loss_weights, generator))
      if FIRSTSTEP:
        print("First batch done!")
        FIRSTSTEP = False
      n_batch_train += 1
    
    ## cross validation testing
    for data_batch in validation_dataset:
      validation_losses = tf.math.add(validation_losses, test_step(model, data_batch, strategy, loss_weights, generator))
      n_batch_valid += 1
  
    ## loss summarization
    training_loss = tf.math.divide(tf.math.reduce_sum(training_losses), n_train)
    validation_loss = tf.math.divide(tf.math.reduce_sum(validation_losses), n_valid)

    ## reports and checkpoint saving
    print('Epoch {}: Average training loss = {}, Validation loss = {}'.format(epoch, training_loss, validation_loss))
    model.save_weights(checkpoint_path)
    print('Time for epoch {} is {} sec, total {} sec, saved.'.format(epoch, time.time()-start, time.time()-allstart))

### for static dataset, probably not being used now
def train_auto(encoder, training_dataset, validation_dataset, epochs, strategy, lr=DEFAULT_LR, restore_checkpoint=True):
  model = encoder.model
  if restore_checkpoint:
    load_checkpoint(model, strategy)
  with strategy.scope():
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                       save_weights_only=True,
                                                       verbose=1)
    model.compile(optimizer = optimizer_type(lr), loss = 'mse', metrics = ['mse'])
    # dataset is batched, no need to re-batch
    model.fit(training_dataset, epochs = epochs, validation_data=validation_dataset, callbacks=[ckpt_callback])
 
