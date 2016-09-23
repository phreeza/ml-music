import os
import time
from glob import glob
import tensorflow as tf

from ops import *

'''
vae implementation, alpha version, used with mnist, but easy to change to work with cifar-10

mnist samples cropped to 26x26 from 28x28 to produce more diverse training examples

LOADS of code sniplets was taken from:

https://jmetzen.github.io/2015-11-27/vae.html
'''

class VAE():
  def __init__(self, batch_size=1,
                z_dim=8, net_size = 384,
                learning_rate = 0.01, keep_prob = 1.0, loss_mode = 1, chunk_samples = 1024):
    """

    Args:
        sess: TensorFlow session
        batch_size: The size of batch. Should be specified before training.
        z_dim: (optional) Dimension of dim for Z. [20]
        net_size: number of nodes in each hidden layer
        keep_prob: dropout keep probability
        loss_mode: 1 -> "L2" or 2 -> "Bournoulli"
    """

    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.z_dim = z_dim
    self.net_size = net_size
    self.keep_prob = keep_prob
    self.loss_mode = loss_mode
    self.chunk_samples = chunk_samples

    self.x_dim = self.chunk_samples
    self.n_points = self.x_dim 

    # tf Graph batch of image (batch_size, height, width, depth)
    self.x_raw = tf.placeholder(tf.float32, [batch_size, self.chunk_samples])

    # distort raw data (decided in the end to leave this task to DataLoader class)
    self.x = self.x_raw

    # Create autoencoder network
    self._create_network()
    # Define loss function based variational upper-bound and
    # corresponding optimizer
    self._create_loss_optimizer()

    self.check = tf.add_check_numerics_ops()
    # Initializing the tensor flow variables
    init = tf.initialize_all_variables()

    # Launch the session
    self.sess = tf.InteractiveSession()
    self.sess.run(init)
    self.saver = tf.train.Saver(tf.all_variables())

  def _create_network(self):


    # Use recognition network to determine mean and
    # (log) variance of Gaussian distribution in latent
    # space
    self.z_mean, self.z_log_sigma_sq = self._recognition_network(self.x)

    # Draw one sample z from Gaussian distribution
    n_z = self.z_dim
    eps = tf.random_normal((self.batch_size, n_z), 0.0, 1.0, dtype=tf.float32)
    # z = mu + sigma*epsilon
    self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(tf.minimum(20.,self.z_log_sigma_sq))), eps))

    # Use generator to determine mean of
    # Bernoulli distribution of reconstructed input
    self.x_reconstr_mean, self.x_reconstr_log_sigma_sq = self._generator_network(self.z)
    self.x_reconstr_stddev = tf.reduce_mean(tf.sqrt(tf.exp(self.x_reconstr_log_sigma_sq)))

  def _recognition_network(self, x):
    # Generate probabilistic encoder (recognition network), which
    # maps inputs onto a normal distribution in latent space.
    # The transformation is parametrized and can be learned.
    x_flatten = tf.reshape(x, [self.batch_size, -1])
    h1 = tf.nn.dropout(tf.nn.softplus(linear(x_flatten, self.net_size, 'q_h0_lin')), self.keep_prob)
    h2 = tf.nn.dropout(tf.nn.softplus(linear(h1, self.net_size, 'q_h1_lin')), self.keep_prob)
    z_mean = linear(tf.reshape(h2, [self.batch_size, -1]), self.z_dim, 'q_h2_lin_mean')
    z_log_sigma_sq = linear(tf.reshape(h2, [self.batch_size, -1]), self.z_dim, 'q_h2_lin_log_sigma_sq')

    return (z_mean, z_log_sigma_sq)

  def _generator_network(self, z):
    # Generate probabilistic decoder (decoder network)
    # The transformation is parametrized and can be learned.

    # project `z` and reshape

    h1 = tf.nn.dropout(tf.nn.softplus(linear(z, self.net_size, 'g_h0_lin')), self.keep_prob)
    h2 = tf.nn.dropout(tf.nn.softplus(linear(h1, self.net_size, 'g_h1_lin')), self.keep_prob)
    x_reconstr_mean = linear(h2, self.n_points, 'g_x_reconstr_mean')
    x_reconstr_log_sigma_sq = linear(h2, self.n_points, 'g_x_reconstr_log_sigma_sq')
    x_reconstr_mean = tf.reshape(x_reconstr_mean, [self.batch_size, self.x_dim])
    x_reconstr_log_sigma_sq = tf.reshape(x_reconstr_log_sigma_sq, [self.batch_size, self.x_dim])
    return x_reconstr_mean, x_reconstr_log_sigma_sq

  def _create_loss_optimizer(self):
    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #     of the input under the reconstructed Bernoulli distribution
    #     induced by the decoder in the data space).
    #     This can be interpreted as the number of "nats" required
    #     for reconstructing the input when the activation in latent
    #     is given.

    orig_energies = tf.reshape(self.x, [self.batch_size, -1])
    new_energies = tf.reshape(self.x_reconstr_mean, [self.batch_size, -1])

    diff = tf.square(tf.sub(orig_energies, new_energies))
    diff_norm = tf.div(diff,tf.exp(tf.minimum(20.,self.x_reconstr_log_sigma_sq)))
    denom_log = tf.log(2*np.pi) + self.x_reconstr_log_sigma_sq
    self.vae_loss_likelihood = tf.reduce_sum(0.5*(diff_norm+denom_log), 1) 

    # 2.) The latent loss, which is defined as the Kullback Leibler divergence
    ##    between the distribution in latent space induced by the encoder on
    #     the data and some prior. This acts as a kind of regularizer.
    #     This can be interpreted as the number of "nats" required
    #     for transmitting the the latent space distribution given
    #     the prior.
    self.vae_loss_kl = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                       - tf.square(self.z_mean)
                                       - tf.exp(tf.minimum(20.,self.z_log_sigma_sq)), 1)

    self.cost = tf.reduce_mean(self.vae_loss_likelihood + 0.5*self.vae_loss_kl) # average over batch

    #self.cost = tf.reduce_mean(self.vae_loss_kl + self.vae_loss_l2)

    self.t_vars = tf.trainable_variables()

    # Use RMSProp optimizer
    opt = tf.train.AdamOptimizer(self.learning_rate) #.minimize(self.cost, var_list=self.t_vars)
    grads, t_vars = zip(*opt.compute_gradients(self.cost, self.t_vars))
    self.gradnorm = tf.global_norm(grads)
    grads = tf.cond(
        tf.global_norm(grads) > 1e-20,
        lambda: tf.clip_by_global_norm(grads, 500.)[0],
        lambda: grads)
    self.optimizer = opt.apply_gradients(zip(grads,t_vars))


  def partial_fit(self, X):
    """Train model based on mini-batch of input data.

    Return cost of mini-batch.
    """

    opt, cost, vae_loss_likelihood, vae_loss_kl, _, gradnorm, stddev = self.sess.run((self.optimizer, self.cost, self.vae_loss_likelihood, self.vae_loss_kl, self.check, self.gradnorm, self.x_reconstr_stddev),
                              feed_dict={self.x_raw: X})
    return cost, vae_loss_likelihood, vae_loss_kl, gradnorm, stddev

  def transform(self, X):
    """Transform data by mapping it into the latent space."""
    # Note: This maps to mean of distribution, we could alternatively
    # sample from Gaussian distribution
    return self.sess.run(self.z_mean, feed_dict={self.x_raw: X})

  def generate(self, z_mu=None):
    """ Generate data by sampling from latent space.

    If z_mu is not None, data for this point in latent space is
    generated. Otherwise, z_mu is drawn from prior in latent
    space.
    """
    if z_mu is None:
        z_mu = np.random.normal(size=(1, self.z_dim))
    # Note: This maps to mean of distribution, we could alternatively
    # sample from Gaussian distribution
    return self.sess.run((self.x_reconstr_mean,self.x_reconstr_log_sigma_sq),
                           feed_dict={self.z: z_mu})

  def reconstruct(self, X):
    """ Use VAE to reconstruct given data. """
    return self.sess.run(self.x_reconstr_mean,
                         feed_dict={self.x_raw: X})

  def encode(self, X):
    """ Use VAE to reconstruct given data. """
    return self.sess.run((self.z_mean,self.z_log_sigma_sq),
                         feed_dict={self.x_raw: X})

  def save_model(self, checkpoint_path, epoch):
    """ saves the model to a file """
    self.saver.save(self.sess, checkpoint_path, global_step = epoch)

  def load_model(self, checkpoint_path):

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print "loading model: ",ckpt.model_checkpoint_path

    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    # use the below line for tensorflow 0.7
    #self.saver.restore(self.sess, ckpt.model_checkpoint_path)
