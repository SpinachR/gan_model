import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import scipy
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


# This function performs a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def generate_z(batch_size, z_dim):
    return np.random.standard_normal([batch_size, z_dim])


def generate_c(batch_size, c_dim):
    return np.random.standard_normal([batch_size, c_dim])


def generate_labels(batch_size, pvals=[0.1]*10):
    return np.random.multinomial(1, pvals, size=batch_size)


# The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
# They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image

    return img


def plot(samples, figsize, image_dim):
    fig = plt.figure(figsize=(figsize, figsize))
    gs = gridspec.GridSpec(figsize, figsize)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(image_dim))

    return fig