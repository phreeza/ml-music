import tensorflow as tf

import numpy as np
import random

class Model():
    def __init__(self, args, infer=False):

        def tf_normal(x, mu, s):
            # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
            norm = tf.sub(x, mu)
            z = tf.reduce_sum(tf.square(tf.div(norm, s)),1)
            result = tf.exp(-z)
            denom = 2*np.pi*tf.reduce_prod(s, 1)
            result = tf.div(result, denom)
            return result

        def get_lossfunc(z_pi, z_mu,  z_sigma, x):
            result = tf_normal(x, z_mu, z_sigma)
            epsilon = 1e-20
            result = tf.mul(result, z_pi)
            result = tf.reduce_sum(result, 1, keep_dims=True)
            result = -tf.log(tf.maximum(result, 1e-20))

            return tf.reduce_sum(result)

        def get_mixture_coef(output):
            z = output
            z_pi = z[:,:self.num_mixture]
            z_mu = z[:,self.num_mixture:(26+1024+1)*self.num_mixture]
            z_sigma = z[:,(26+1024+1)*self.num_mixture:]

            # apply transformations
            z_pi = tf.nn.softmax(z_pi)
            z_sigma = tf.exp(z_sigma)

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

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 1024 + 26], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 1024 + 26],name = 'target_data')
        self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        self.num_mixture = args.num_mixture

        # 
        NOUT = self.num_mixture * (1 + 2*(1024 + 26))

        with tf.variable_scope('rnnlm'):
            output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
            output_b = tf.get_variable("output_b", [NOUT])

        #inputs = tf.split(1, args.seq_length, self.input_data)
        #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        inputs = tf.unpack(tf.transpose(self.input_data, perm=(1,0,2)))
        print self.input_data
        print inputs[0]

        outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        self.final_state = last_state

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.target_data,[-1, 1024 + 26])

        [o_pi, o_mu, o_sigma] = get_mixture_coef(output)

        self.pi = o_pi
        self.mu = o_mu
        self.sigma = o_sigma

        lossfunc = get_lossfunc(o_pi, o_mu, o_sigma, flat_target_data)
        self.cost = lossfunc / (args.batch_size * args.seq_length)

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
