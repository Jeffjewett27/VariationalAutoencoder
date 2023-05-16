import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import data
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from datetime import datetime

from tensorflow.python.eager.def_function import run_functions_eagerly
from cvae import CVAE

from vae import VAE, VAESampling
from mnist_constants import *
from face_process import get_face_data

tf.executing_eagerly()

def get_face_encoder():
  latent_dim = 48

  encoder_inputs = keras.Input(shape=(128, 128, 3))

  m1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
  m1 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(m1)
  m1 = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(m1)
  m1 = layers.Flatten()(m1)
  m1 = layers.Dense(64, activation="relu")(m1)

  x = layers.Dense(64, activation="relu")(m1)
  x = layers.Dense(64, activation="relu")(x)
  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
  z = VAESampling()([z_mean, z_log_var])
  encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
  return encoder

def get_face_decoder():
  latent_dim = 48

  latent_inputs = keras.Input(shape=(latent_dim,), name="image")

  x = layers.Dense(16 * 16 * 64 * 3, activation="relu", name="decoder_dense_1")(latent_inputs)
  x = layers.Reshape((16, 16, 64 * 3))(x)
  x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  print(decoder.summary())
  return decoder

def get_face_conditional_encoder():
  latent_dim = 12

  encoder_inputs = keras.Input(shape=(128, 128, 3))
  gender_input = keras.Input(shape=(1,), name="gender")
  age_input = keras.Input(shape=(1,), name="age")
  race_input = keras.Input(shape=(5,), name="race")

  m1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
  m1 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(m1)
  m1 = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(m1)
  m1 = layers.Flatten()(m1)
  m1 = layers.Dense(16, activation="relu")(m1)
  m1 = keras.Model(inputs=encoder_inputs, outputs=m1)

  m2 = layers.concatenate([gender_input, age_input, race_input])
  m2 = layers.Dense(32, activation="relu")(m2)
  m2 = layers.Dense(16, activation="relu")(m2)
  m2 = keras.Model(inputs=[gender_input, age_input, race_input], outputs=m2)

  x = layers.concatenate([m1.output, m2.output])
  x = layers.Dense(32, activation="relu")(x)
  x = layers.Dense(16, activation="relu")(x)
  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
  z = VAESampling()([z_mean, z_log_var])
  encoder = keras.Model([m1.input, m2.input], [z_mean, z_log_var, z], name="encoder")
  return encoder

def get_face_conditional_decoder():
  latent_dim = 12

  latent_inputs = keras.Input(shape=(latent_dim,), name="image")
  gender_input = keras.Input(shape=(1,), name="gender")
  age_input = keras.Input(shape=(1,), name="age")
  race_input = keras.Input(shape=(5,), name="race")

  x = layers.concatenate([latent_inputs, gender_input, age_input, race_input])
  x = layers.Dense(16 * 16 * 64 * 3, activation="relu", name="decoder_dense_1")(x)
  x = layers.Reshape((16, 16, 64 * 3))(x)
  x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
  decoder = keras.Model([latent_inputs, gender_input, age_input, race_input], decoder_outputs, name="decoder")
  print(decoder.summary())
  return decoder

def get_face_vae():
  encoder = get_face_encoder()
  decoder = get_face_decoder()
  vae = VAE(encoder, decoder)
  return vae

def get_face_cvae():
  encoder = get_face_conditional_encoder()
  decoder = get_face_conditional_decoder()
  vae = CVAE(encoder, decoder)
  return vae

def train_vae(vae, data, callbacks, prefix):
  vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
  vae.fit(data, epochs=100, batch_size=8, callbacks=callbacks)
  vae.save_weights(os.path.join(directory, prefix + model_template.format(datestr)))

if __name__ == "__main__":
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)

  dataset = get_face_data()
  datestr = datetime.now().strftime('%m-%d-%H-%M')

  prefix = "face-"
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(directory, 'checkpoints', prefix + datestr + '-epoch-{epoch:02d}.hdf5'),
      save_weights_only=True,
      monitor='loss',
      mode='min',
      save_best_only=True)

  vae = get_face_vae()
  train_vae(vae, dataset.batch(32), [model_checkpoint_callback], prefix)
