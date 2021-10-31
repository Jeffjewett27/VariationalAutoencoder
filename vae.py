import imp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class VAESampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# def log_normal_pdf(sample, mean, logvar, raxis=1):
#   log2pi = tf.math.log(2. * np.pi)
#   return tf.reduce_sum(
#       -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#       axis=raxis)

# def compute_loss(model, x):
#   mean, logvar = model.encode(x)
#   z = model.reparameterize(mean, logvar)
#   x_logit = model.decode(z)
#   cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
#   logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#   logpz = log_normal_pdf(z, 0., 0.)
#   logqz_x = log_normal_pdf(z, mean, logvar)
#   return -tf.reduce_mean(logpx_z + logpz - logqz_x)

# @tf.function
# def train_step(model, x, optimizer):
#   """Executes one training step and returns the loss.

#   This function computes the loss and gradients, and uses the latter to
#   update the model's parameters.
#   """
#   with tf.GradientTape() as tape:
#     loss = compute_loss(model, x)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))