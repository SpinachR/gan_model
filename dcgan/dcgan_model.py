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


def _disc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=utils.lrelu,
                        outputs_collections=outputs_collections,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        kernel_size=[5, 5], stride=2, padding='SAME') as arg_scp:
        return arg_scp


def discriminator(image, df_dim=64, reuse=None, is_training=True, scope='discriminator'):
    '''discriminator
    Args:
        df_dim: (optional) Dimension of D filters in first conv layer. [64]
    '''
    with tf.variable_scope(scope, default_name='discriminator', reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_disc_arg_scope(is_training, end_pts_collection)):
            h0 = slim.conv2d(image, df_dim, scope='d_h0_conv')
            h1 = slim.conv2d(h0, df_dim*2, scope='d_h1_conv')
            h2 = slim.conv2d(h1, df_dim*4, scope='d_h2_conv')
            h3 = slim.conv2d(h2, df_dim*8, scope='d_h3_conv')
            h4 = slim.fully_connected(slim.flatten(h3), 1, activation_fn=None, scope='d_h4_lin')
            return tf.nn.sigmoid(h4), h4


def _gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=_batch_norm_params(is_training),
                            kernel_size=[5, 5], stride=2, padding="SAME") as arg_scp:

            return arg_scp


def generator(z, output_h, output_w, output_d, y=None, gf_dim=64, is_training=True, scope='generator'):
    '''generator
    Args:
        z: noise input
        output_h: x's height
        output_w: x's width
        output_d: x's depth (if x is grey image, output_d=1)
        gf_dim: Dimension of G filters in last conv layer. [64]
    '''

    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_gen_arg_scope(is_training, end_pts_collection)):
            s_h, s_w = output_h, output_w
            s_h2, s_w2 = utils.conv_out_size_same(s_h, 2), utils.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = utils.conv_out_size_same(s_h2, 2), utils.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = utils.conv_out_size_same(s_h4, 2), utils.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = utils.conv_out_size_same(s_h8, 2), utils.conv_out_size_same(s_w8, 2)

            h0 = slim.fully_connected(z, gf_dim*8*s_h16*s_w16, activation_fn=None, scope='g_h0_lin')
            h0 = tf.reshape(h0, [-1, s_h16, s_w16, gf_dim*8])
            h1 = tf.nn.relu(slim.batch_norm(h0, scope='g_h1_batchnorm', **_batch_norm_params(is_training)))
            h2 = slim.conv2d_transpose(h1, gf_dim*4, scope='g_h2_con2d_transpose')
            h3 = slim.conv2d_transpose(h2, gf_dim*2, scope='g_h3_con2d_transpose')
            h4 = slim.conv2d_transpose(h3, gf_dim, scope='g_h4_con2d_transpose')
            h5 = slim.conv2d_transpose(h4, output_d, scope='g_h5_con2d_transpose')

            return tf.nn.tanh(h5)

