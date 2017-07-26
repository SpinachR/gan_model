'''
paper: improved Techniques for Training GAN
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import utils


def _batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def gaussian_noise_layer(input_layer, sigma, is_training=True):
    if is_training is True:
        input_layer = input_layer + tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=sigma, dtype=tf.float32)
    return input_layer


def _gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=utils.lrelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=[4, 4], stride=2, padding="SAME") as arg_scp:

            return arg_scp


def generator(z, x_dim, c=None, labels=None, is_training=True, scope='generator'):
    '''
    :param z: gaussian noise
    :param c: latent variables
    :param labels: y variables (one-hot vector)

    :return: x (batch_size * x_dim)
    '''
    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_gen_arg_scope(is_training, end_pts_collection)):
            inputs = z
            if c is not None:
                inputs = tf.concat([inputs, c], axis=1)
            if labels is not None:
                inputs = tf.concat([inputs, labels], axis=1)

            net = slim.fully_connected(inputs, 500, scope='g_fc1')
            net = slim.fully_connected(net, 500, scope='g_fc2')
            net = slim.fully_connected(net, x_dim,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       activation_fn=tf.sigmoid,
                                       scope='g_output')
            #print('g: ', net.get_shape())
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return net, end_pts


def _disc_arg_scope(outputs_collections=None):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        outputs_collections=outputs_collections) as arg_scp:
        return arg_scp


def discriminator(x, output_dim, is_training=True, reuse=None, scope='discriminator'):
    with tf.variable_scope(scope, default_name='discriminator', reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_disc_arg_scope(end_pts_collection)):
            net = gaussian_noise_layer(x, sigma=0.3, is_training=is_training)
            net = slim.fully_connected(net, 1000, scope='d_fc1')

            net = gaussian_noise_layer(net, sigma=0.5, is_training=is_training)
            net = slim.fully_connected(net, 500, scope='d_fc2')

            net = gaussian_noise_layer(net, sigma=0.5, is_training=is_training)
            net = slim.fully_connected(net, 250, scope='d_fc3')

            net = gaussian_noise_layer(net, sigma=0.5, is_training=is_training)
            net = slim.fully_connected(net, 250, scope='d_fc4')

            net = gaussian_noise_layer(net, sigma=0.5, is_training=is_training)
            feature = slim.fully_connected(net, 250, scope='d_fc5')

            net = gaussian_noise_layer(feature, sigma=0.5, is_training=is_training)
            net = slim.fully_connected(net, output_dim, activation_fn=None, scope='d_output')

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return net, feature, end_pts