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

@tf.function
def test_step(model, data_batch, strategy):
  images = data_batch[0]
  latents = data_batch[1]
  def true_step(images, latents):
    logits = model(images, training=False)
    batch_loss = losses.MSE(latents, logits)
    return batch_loss

  if isColab():
    batch_loss = strategy.run(true_step, args=(images, latents,))       # make sure this is a tuple using ','
  else:
    batch_loss = true_step(images, latents)
  return strategy.reduce(tf.distribute.ReduceOp.SUM, batch_loss, axis=None)

@tf.function
def train_step(model, data_batch, optimizer, strategy):
  images = data_batch[0]
  latents = data_batch[1]
  def true_step(images, latents):
    with tf.GradientTape() as tape:
      logits = model(images, training=True)
      batch_loss = losses.MSE(latents, logits)
    grads = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return batch_loss

  if isColab():
    batch_loss = strategy.run(true_step, args=(images, latents,))       # make sure this is a tuple using ','
  else:
    batch_loss = true_step(images, latents)
  return strategy.reduce(tf.distribute.ReduceOp.SUM, batch_loss, axis=None)


def train(encoder, generator, dataset_gen_func, n_train, n_valid, epochs, strategy, model_type=None, lr=DEFAULT_LR, restore_checkpoint=True):
  model = encoder.model
  global FIRSTSTEP

  if restore_checkpoint:
    load_checkpoint(model, strategy)
  ckpt = tf.train.Checkpoint(model=model)

  optimizer = optimizer_type(lr)

  allstart = time.time()
  for epoch in range(1, epochs+1):
    start = time.time()
    training_dataset, validation_dataset = dataset_gen_func(generator, strategy, batch=True, n_train=n_train, n_valid=n_valid, model_type=model_type)
    training_losses = 0.0
    validation_losses = 0.0
    for data_batch in training_dataset:
      training_losses = tf.math.add(training_losses, train_step(model, data_batch, optimizer, strategy))
      if FIRSTSTEP:
        print("First batch done!")
        FIRSTSTEP = False
    for data_batch in validation_dataset:
      validation_losses += test_step(model, data_batch, strategy)
    training_loss = tf.math.divide(tf.math.reduce_sum(training_losses), n_train)
    validation_loss = tf.math.divide(tf.math.reduce_sum(validation_losses), n_valid)
    print('Epoch {}: Average training loss = {}, Validation loss = {}'.format(epoch, training_loss, validation_loss))
    ckpt.save(checkpoint_path)
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
 
