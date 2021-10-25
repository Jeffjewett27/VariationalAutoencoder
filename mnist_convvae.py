import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from datetime import datetime
from cvae import CVAE

from vae import VAE, VAESampling
from mnist_constants import *

class MnistConvVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(MnistConvVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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

def get_mnist_data():
  (x_train, l_train), (x_test, l_test) = keras.datasets.mnist.load_data()
  y_train = np.zeros((l_train.shape[0], l_train.max()+1), dtype=np.float32)
  y_train[np.arange(l_train.shape[0]), l_train] = 1
  y_test = np.zeros((l_test.shape[0], l_test.max()+1), dtype=np.float32)
  y_test[np.arange(l_test.shape[0]), l_test] = 1
  mnist_digits = np.concatenate([x_train, x_test], axis=0)
  mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
  mnist_labels = np.concatenate([y_train, y_test], axis=0)
  return mnist_digits, mnist_labels

if __name__ == "__main__":
  # encoder = get_mnist_conditional_encoder(2)
  # decoder = get_mnist_conditional_decoder(2)
  encoder = get_mnist_encoder(2)
  decoder = get_mnist_decoder(2)
  mnist_digits, mnist_labels = get_mnist_data()

  datestr = datetime.now().strftime('%m-%d-%H-%M')

  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(directory, 'checkpoints', datestr + '-epoch-{epoch:02d}.hdf5'),
      save_weights_only=True,
      monitor='loss',
      mode='min',
      save_best_only=True)

  vae = VAE(encoder, decoder)
  vae.compile(optimizer=keras.optimizers.Adam())
  #vae.fit([mnist_digits, mnist_labels], epochs=1, batch_size=128, callbacks=[model_checkpoint_callback])
  vae.fit(mnist_digits, epochs=100, batch_size=128, callbacks=[model_checkpoint_callback])

  vae.save_weights(os.path.join(directory, model_template.format(datestr)))