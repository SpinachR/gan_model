import matplotlib
matplotlib.use('Agg')
import os
import time
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import improved_gan_model as model
from utils import utils
from tensorflow.examples.tutorials.mnist import input_data


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.mnist = input_data.read_data_sets('../MNIST')

        self.build_model(config)

        if not os.path.exists(config.sampledir):
            os.makedirs(config.sampledir)

        if not os.path.exists(config.checkpoint_basename):
            os.makedirs(config.checkpoint_basename)

        if not os.path.exists(config.logdir):
            os.makedirs(config.logdir)

        self.loss_summaries = tf.summary.merge([
            tf.summary.scalar("loss_D_real", self.loss_lab),
            tf.summary.scalar("loss_D_fake", self.loss_unl),
            tf.summary.scalar("loss_D", self.loss_gen),
            tf.summary.scalar("loss_G", self.accuracy),
        ])

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(config.logdir)
        self.sess = tf.Session()

    def build_model(self, config):
        self.x_lab = tf.placeholder(tf.float32, [None, config.x_dim], name='x_lab')
        self.y = tf.placeholder(tf.int64, [None], name='y')
        self.x_unl = tf.placeholder(tf.float32, [None, config.x_dim], name='x_unl')
        self.z = tf.placeholder(tf.float32, [None, config.z_dim], name='z')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        output_before_softmax_lab, _, _ = model.discriminator(self.x_lab, config.d_output_dim, is_training=self.is_training)
        output_before_softmax_unl, unl_feature, _ = model.discriminator(self.x_unl, config.d_output_dim, is_training=self.is_training, reuse=True)
        self.fake, _ = model.generator(self.z, config.x_dim, is_training=self.is_training)
        output_before_softmax_fake, fake_feature, _ = model.discriminator(self.fake, config.d_output_dim, is_training=self.is_training, reuse=True)

        with tf.variable_scope("Loss_supervised"):
            self.loss_lab = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=output_before_softmax_lab)

        with tf.variable_scope("Loss_unsupervised"):
            loss_real = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(output_before_softmax_unl), logits=output_before_softmax_unl)
            loss_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(output_before_softmax_fake), logits=output_before_softmax_fake)
            self.loss_unl = loss_fake + loss_real

        with tf.variable_scope("Loss_gen"):
            mom_gen = tf.reduce_mean(fake_feature, axis=0)
            mom_real = tf.reduce_mean(unl_feature, axis=0)
            self.loss_gen = tf.reduce_mean(tf.square(mom_gen - mom_real))

        with tf.variable_scope("Optimizer_D"):
            vars_D = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
            self.opt_loss_lab = tf.train.AdamOptimizer(config.lr_d, beta1=config.beta1).minimize(self.loss_lab, var_list=vars_D)
            self.opt_loss_unl = tf.train.AdamOptimizer(config.lr_d, beta1=config.beta1).minimize(self.loss_unl, var_list=vars_D)

        with tf.variable_scope("Optimizer_G"):
            vars_G = [var for var in tf.trainable_variables() if 'generator' in var.name]
            self.opt_G = tf.train.AdamOptimizer(config.lr_g, beta1=config.beta1).minimize(self.loss_gen, var_list=vars_G)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_before_softmax_lab, 1), self.y), tf.float32))

    def fit(self):
        config = self.config
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(config.max_steps):
                t1 = time.time()
                z = utils.generate_z(config.unlabeled_batch, config.z_dim)
                x_unl, dummy = self.mnist.train.next_batch(config.unlabeled_batch)

                x_lab, y = self.mnist.train.next_batch(config.labeled_batch)

                _, _, cur_loss_lab, cur_loss_unl = sess.run(
                    [self.opt_loss_lab, self.opt_loss_unl, self.loss_lab, self.loss_unl],
                    feed_dict={self.z: z, self.x_lab: x_lab, self.x_unl: x_unl, self.y: y, self.is_training: True}
                )

                _, cur_loss_gen = sess.run(
                    [self.opt_G, self.loss_gen],
                    feed_dict={self.z: z, self.x_unl: x_unl, self.is_training: True}
                )
                t2 = time.time()

                if (step+1) % config.summary_every_n_steps == 0:
                    summary_feed_dict = {
                        self.z: z, self.x_lab: x_lab, self.x_unl: x_unl, self.y: y, self.is_training: False
                    }
                    self.make_summary(summary_feed_dict, step+1)

                if (step+1) % config.sample_every_n_steps == 0:
                    eta = (t2 - t1) * (config.max_steps - step + 1)
                    print("Finished {}/{} step, ETA:{:.2f}s"
                          .format(step + 1, config.max_steps, eta))

                    z_ = utils.generate_z(25, config.z_dim)
                    samples = sess.run(self.fake, feed_dict={self.z: z_, self.is_training: False})
                    fig = utils.plot(samples, 5)
                    plt.savefig(config.sampledir + '/{}.png'.format(step+1), bbox_inches='tight')
                    plt.close(fig)

                if (step + 1) % config.savemodel_every_n_steps == 0:
                    self.saver.save(sess, config.checkpoint_basename+"/model.ckpt", global_step=step + 1)

    def make_summary(self, feed_dict, step):
        summary, cur_accuracy = self.sess.run([self.loss_summaries, self.accuracy], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step=step)
        print("{} steps training accuracy: {}".format(step, cur_accuracy))


