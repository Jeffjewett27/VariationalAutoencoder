import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
from datetime import datetime
from cvae import CVAE

from vae import VAE, VAESampling
from mnist_constants import *
from mnist_data import get_mnist_data

def get_mnist_encoder(latent_dim):
  encoder_inputs = keras.Input(shape=(28, 28, 1))
  x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
  x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Flatten()(x)
  x = layers.Dense(16, activation="relu")(x)
  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
  z = VAESampling()([z_mean, z_log_var])
  encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
  return encoder

def get_mnist_decoder(latent_dim):
  latent_inputs = keras.Input(shape=(latent_dim,))
  x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
  x = layers.Reshape((7, 7, 64))(x)
  x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  return decoder

def get_mnist_conditional_encoder(latent_dim):
  encoder_inputs = keras.Input(shape=(28, 28, 1))
  digit_inputs = keras.Input(shape=(10,))

  m1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
  m1 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(m1)
  m1 = layers.Flatten()(m1)
  m1 = layers.Dense(16, activation="relu")(m1)
  m1 = keras.Model(inputs=encoder_inputs, outputs=m1)

  m2 = layers.Dense(16, activation="relu")(digit_inputs)
  m2 = keras.Model(inputs=digit_inputs, outputs=m2)

  x = layers.concatenate([m1.output, m2.output])
  x = layers.Dense(16, activation="relu")(x)
  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
  z = VAESampling()([z_mean, z_log_var])
  encoder = keras.Model([m1.input, m2.input], [z_mean, z_log_var, z], name="encoder")
  return encoder

def get_mnist_conditional_decoder(latent_dim):
  latent_inputs = keras.Input(shape=(latent_dim,))
  digit_inputs = keras.Input(shape=(10,))
  x = layers.concatenate([latent_inputs, digit_inputs])
  x = layers.Dense(7 * 7 * 64, activation="relu", name="decoder_dense_1")(x)
  x = layers.Reshape((7, 7, 64))(x)
  x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
  decoder = keras.Model([latent_inputs, digit_inputs], decoder_outputs, name="decoder")
  return decoder

def get_mnist_vae():
  encoder = get_mnist_encoder(2)
  decoder = get_mnist_decoder(2)
  vae = VAE(encoder, decoder)
  return vae

def get_mnist_cvae():
  encoder = get_mnist_conditional_encoder(2)
  decoder = get_mnist_conditional_decoder(2)
  vae = CVAE(encoder, decoder)
  return vae

def train_vae(vae, data, callbacks, prefix):
  vae.compile(optimizer=keras.optimizers.Adam())
  vae.fit(data, epochs=100, batch_size=128, callbacks=callbacks)
  vae.save_weights(os.path.join(directory, prefix + model_template.format(datestr)))

if __name__ == "__main__":
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)

  iscvae = sys.argv[1] == "cvae"

  mnist_digits, mnist_labels = get_mnist_data()
  datestr = datetime.now().strftime('%m-%d-%H-%M')

  print([mnist_digits, mnist_labels])

  prefix = "cvae-" if iscvae else "vae-"
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(directory, 'checkpoints', prefix + datestr + '-epoch-{epoch:02d}.hdf5'),
      save_weights_only=True,
      monitor='loss',
      mode='min',
      save_best_only=True)

  if iscvae:
    vae = get_mnist_cvae()
    train_vae(vae, [mnist_digits, mnist_labels], [model_checkpoint_callback], prefix)
  else:
    vae = get_mnist_vae()
    train_vae(vae, mnist_digits, [model_checkpoint_callback], prefix)
