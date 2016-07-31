import tensorflow as tf

import numpy as np
import random

class Model():
    def __init__(self, args, infer=False):

        def tf_normal(x, mu, s):
            # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
            norm = tf.sub(x, mu)
            #tf.histogram_summary('z-score', tf.div(norm,tf.sqrt(s)))
            #tf.histogram_summary('std-dev', tf.sqrt(s))
            z = tf.reduce_sum(tf.div(tf.square(norm), s),1)
            denom_log = tf.reduce_sum(tf.log(tf.sqrt(2*np.pi*s)), 1,name="denom_log")
            result = -z/2-denom_log
            return result

        def get_lossfunc(z_pi, z_mu,  z_sigma, x):
            result = -tf_normal(x, z_mu, z_sigma)
            #result = tf.mul(result, z_pi)
            #result = tf.reduce_sum(result, 1, keep_dims=True)
            #result = -tf.log(tf.maximum(result, 1e-30))

            return tf.reduce_sum(result)

        def get_mixture_coef(output):
            z = output
            z_pi = z[:,:self.num_mixture]
            z_mu = z[:,self.num_mixture:(args.chunk_samples+1)*self.num_mixture]
            z_sigma = z[:,(args.chunk_samples+1)*self.num_mixture:]

            # apply transformations
            z_pi = tf.nn.softmax(z_pi, name='z_pi')
            z_sigma = tf.exp(z_sigma, name='z_sigma')

            return [z_pi, z_mu, z_sigma]

        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)

        if (infer == False and args.keep_prob < 1): # training mode
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = args.keep_prob)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, args.chunk_samples], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, args.chunk_samples],name = 'target_data')
        self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        self.num_mixture = args.num_mixture

        # 
        NOUT = self.num_mixture * (1 + 2*(args.chunk_samples))

        with tf.variable_scope('rnnlm'):
            output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
            output_b = tf.get_variable("output_b", [NOUT])

        #inputs = tf.split(1, args.seq_length, self.input_data)
        #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        inputs = tf.unpack(tf.transpose(self.input_data, perm=(1,0,2)))

        outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        self.final_state = last_state

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.target_data,[-1, args.chunk_samples])

        [o_pi, o_mu, o_sigma] = get_mixture_coef(output)

        self.pi = o_pi
        self.mu = o_mu
        self.sigma = o_sigma

        lossfunc = get_lossfunc(o_pi, o_mu, o_sigma, flat_target_data)
        self.cost = lossfunc / (args.batch_size * args.seq_length)
        tf.scalar_summary('cost', self.cost)


        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        grads = tf.cond(
            tf.global_norm(grads) > 1e-20,
            lambda: tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)[0],
            lambda: grads)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, args, num=1200):
        def sample_gaussian(mu, sigma):
            return mu + (sigma*np.random.randn(sigma.shape))

        prev_x = np.randn((1, 1, args.chunk_samples), dtype=np.float32)
        prev_state = sess.run(self.cell.zero_state(1, tf.float32))

        chunks = np.zeros((num, args.chunk_samples), dtype=np.float32)

        for i in xrange(num):

            feed = {self.input_data: prev_x, self.initial_state:prev_state}

            [o_pi, o_mu, o_sigma, next_state] = sess.run([self.pi, self.mu, self.sigma, self.final_state],feed)

            idx = get_pi_idx(random.random(), o_pi[0])

            next_x = sample_gaussian(o_mu, o_sigma)
            chunks[i] = next_x

            prev_x = np.zeros((1, 1, args.chunk_samples), dtype=np.float32)
            prev_x[0][0] = next_x
            prev_state = next_state

        return chunks

