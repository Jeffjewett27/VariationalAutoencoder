import sys
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from cvae import CVAE
from vae import VAE
from mnist_convvae import get_mnist_cvae, get_mnist_vae

def display_image(epoch_no):
 return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

#plt.imshow(display_image(epoch))
#plt.axis('off')  # Display images

import matplotlib.pyplot as plt


def plot_latent_space(vae, iscond=False, label=0, n=30, figsize=15):
    label_oh = tf.one_hot([label], 10)
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.5
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)[::-1]
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            if (iscond):
                z_sample = tf.constant([[xi, yi]])
                x_decoded = vae.decoder([z_sample, label_oh])
            else:
                z_sample = np.array([[xi, yi]])
                x_decoded = vae.decoder(z_sample)
            digit = x_decoded[0].numpy().reshape(digit_size, digit_size)
            figure[ 
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

# cvae 'outputs/mnist_vae/10-24-22-06.h5'
# vae 'outputs/mnist_vae/10-25-07-07.h5'

if __name__ == "__main__":
    iscvae = sys.argv[1] == "cvae"
    model = sys.argv[2]

    if iscvae:
        vae = get_mnist_cvae()
        vae.built = True
        vae.load_weights(model)
        plot_latent_space(vae, True, int(sys.argv[3]))
    else:
        vae = get_mnist_vae()
        vae.built = True
        vae.load_weights(model)
        plot_latent_space(vae, False)