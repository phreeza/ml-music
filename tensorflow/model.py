import tensorflow as tf

import numpy as np
import random

class Model():
    def __init__(self, args, sample=False):

        def tf_normal(x, mu, s, rho):
            x = tf.expand_dims(x,2)
            norm = tf.sub(x[:,:args.chunk_samples,:], mu)
            #tf.histogram_summary('z-score', tf.div(norm,tf.sqrt(s)))
            #tf.histogram_summary('std-dev', tf.sqrt(s))
            z = tf.div(tf.square(norm), s)
            denom_log = tf.log(tf.sqrt(2*np.pi*s))
            result = tf.reduce_sum(-z/2-denom_log + 
                                   (tf.log(rho)*(1+x[:,args.chunk_samples:,:])
                                    +tf.log(1-rho)*(1-x[:,args.chunk_samples:,:]))/2, 1) 

            return result

        def get_lossfunc(z_pi, z_mu,  z_sigma, z_rho, x):
            normals = tf_normal(x, z_mu, z_sigma, z_rho)
            result = -tf_logsumexp(tf.log(z_pi)+normals)
            return tf.reduce_sum(result)
        
        def tf_logsumexp(x):
            max_val = tf.reduce_max(x,1, keep_dims=True) 
            ret = tf.log(tf.reduce_sum(tf.exp(x - max_val), 1, keep_dims=True)) + max_val
            return ret

        def get_mixture_coef(output):
            z = output
            z_pi = z[:,:self.num_mixture]
            z_mu = tf.reshape(z[:,self.num_mixture:(args.chunk_samples+1)*self.num_mixture],[-1,args.chunk_samples,self.num_mixture])
            z_sigma = tf.reshape(z[:,(args.chunk_samples+1)*self.num_mixture:(2*args.chunk_samples+1)*self.num_mixture],[-1,args.chunk_samples,self.num_mixture])
            z_rho = tf.reshape(z[:,(2*args.chunk_samples+1)*self.num_mixture:],[-1,args.chunk_samples,self.num_mixture])
            
            # apply transformations

            #softmax with lower bound
            z_pi = (tf.nn.softmax(z_pi, name='z_pi')+0.01)/(1.+0.01*args.num_mixture)
            z_sigma = tf.exp(z_sigma, name='z_sigma')
            z_rho = tf.sigmoid(z_rho, name='z_rho')

            return [z_pi, z_mu, z_sigma, z_rho]

        self.args = args
        if sample:
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

        if (sample == False and args.keep_prob < 1): # training mode
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = args.keep_prob)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples],name = 'target_data')
        self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        self.num_mixture = args.num_mixture

        # 
        NOUT = self.num_mixture * (1 + 3*(args.chunk_samples))

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
        flat_target_data = tf.reshape(self.target_data,[-1, 2*args.chunk_samples])

        [o_pi, o_mu, o_sigma, o_rho] = get_mixture_coef(output)

        self.pi = o_pi
        self.mu = o_mu
        self.sigma = o_sigma
        self.rho = o_rho

        lossfunc = get_lossfunc(o_pi, o_mu, o_sigma, o_rho, flat_target_data)
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

    def sample(self, sess, args, num=1200, start=None):

        def sample_gaussian(mu, sigma):
            return mu + (sigma*np.random.randn(*sigma.shape))

        if start is None:
            prev_x = np.random.randn(1, 1, 2*args.chunk_samples)
        else:
            prev_x = start[np.newaxis,np.newaxis,:]
        prev_state = sess.run(self.cell.zero_state(1, tf.float32))

        chunks = np.zeros((num, 2*args.chunk_samples), dtype=np.float32)
        mus = np.zeros((num, args.chunk_samples), dtype=np.float32)
        sigmas = np.zeros((num, args.chunk_samples), dtype=np.float32)
        pis = np.zeros((num, args.num_mixture), dtype=np.float32)
        
        for i in xrange(num):
            feed = {self.input_data: prev_x, self.initial_state:prev_state}
            [o_pi, o_mu, o_sigma, o_rho, next_state] = sess.run([self.pi, self.mu, self.sigma, self.rho, self.final_state],feed)
            p = o_pi[0]
            p = (p-p.min())
            p = p/p.sum()
            idx = np.random.choice(range(self.num_mixture),p = p)
            next_x = np.hstack((sample_gaussian(o_mu[:,:,idx], o_sigma[:,:,idx]),
                     2.*(o_rho[:,:,idx] > np.random.random(o_rho.shape[:2]))-1.))
            chunks[i] = next_x
            mus[i] = o_mu[:,:,idx]
            sigmas[i] = o_sigma[:,:,idx]
            pis[i] = p

            prev_x = np.zeros((1, 1, 2*args.chunk_samples), dtype=np.float32)
            prev_x[0][0] = next_x
            prev_state = next_state

        return chunks, mus, sigmas, pis
