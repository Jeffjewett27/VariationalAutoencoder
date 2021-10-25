import tensorflow as tf
import numpy as np

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

def get_mnist_data():
  (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

  train_images = preprocess_images(train_images)
  test_images = preprocess_images(test_images)

  train_size = 60000
  batch_size = 32
  test_size = 10000

  train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                  .shuffle(train_size).batch(batch_size))
  test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                  .shuffle(test_size).batch(batch_size))

  return train_dataset, test_dataset