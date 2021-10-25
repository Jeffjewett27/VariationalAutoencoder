train_size = 60000
batch_size = 32
test_size = 10000
epochs = 10
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16

directory = 'outputs/mnist_vae'
image_dir = 'images'
image_template = 'mnist_sample_{:04d}.png'
model_template = '{0}.h5'