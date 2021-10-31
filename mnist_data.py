import tensorflow as tf
from tensorflow import keras
import numpy as np

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