import tensorflow as tf

import numpy as np
import random

class Model():
    def __init__(self, args, sample=False):

        def tf_normal(x, mu, s, rho):
            with tf.variable_scope('normal'):
                x = tf.expand_dims(x,2)
                norm = tf.sub(x[:,:args.chunk_samples,:], mu)
                z = tf.div(tf.square(norm), s)
                tf.histogram_summary('z-score', tf.div(norm,tf.sqrt(s)))
                tf.histogram_summary('std-dev', tf.sqrt(s))
                tf.scalar_summary('std-dev-mean', tf.reduce_mean(tf.sqrt(s)))
                denom_log = tf.log(tf.maximum(1e-20,tf.sqrt(2*np.pi*s)),name='denom_log')
                result = tf.reduce_sum(-z/2-10*denom_log + 
                                       (tf.log(rho,name='log_rho')*(1+x[:,args.chunk_samples:,:])
                                        +tf.log(tf.maximum(1e-20,1-rho),name='log_rho_inv')*(1-x[:,args.chunk_samples:,:]))/2, 1) 

            return result

        def get_lossfunc(z_pi, z_mu,  z_sigma, z_rho, x):
            normals = tf_normal(x, z_mu, z_sigma, z_rho)
            result = -tf_logsumexp(tf.log(tf.maximum(1e-20,z_pi))+normals)

            return tf.reduce_sum(result)
        
        def tf_logsumexp(x):
            with tf.variable_scope('logsumexp'):
                max_val = tf.reduce_max(x,1, keep_dims=True) 
                ret = tf.log(tf.reduce_sum(tf.exp(x - max_val), 1, keep_dims=True)) + max_val
                return ret

        def get_mixture_coef(output):
            with tf.variable_scope('get_mixture'):
                z = output
                z_pi = z[:,:self.num_mixture]
                z_mu = tf.reshape(z[:,self.num_mixture:(args.chunk_samples+1)*self.num_mixture],[-1,args.chunk_samples,self.num_mixture],name='z_mu')
                z_sigma = tf.reshape(z[:,(args.chunk_samples+1)*self.num_mixture:(2*args.chunk_samples+1)*self.num_mixture],[-1,args.chunk_samples,self.num_mixture])
                z_rho = tf.reshape(z[:,(2*args.chunk_samples+1)*self.num_mixture:],[-1,args.chunk_samples,self.num_mixture])
                
                # apply transformations

                #softmax with lower bound
                #z_pi = (tf.nn.softmax(z_pi, name='z_pi')+0.01)/(1.+0.01*args.num_mixture)
                z_pi = tf.nn.softmax(z_pi, name='z_pi')
                z_sigma = tf.exp(z_sigma, name='z_sigma')
                z_rho = tf.maximum(1e-20,tf.sigmoid(z_rho, name='z_rho'))

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

        output_w = tf.Variable(tf.random_normal([args.rnn_size, NOUT],stddev=0.2), name="output_w")
        output_b = tf.Variable(tf.zeros([NOUT]), name="output_b")

        #inputs = tf.split(1, args.seq_length, self.input_data)
        #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        #inputs = tf.unpack(tf.transpose(self.input_data, perm=(1,0,2)))

        # input shape: (batch_size, n_steps, n_input)
        inputs = tf.transpose(self.input_data, [1, 0, 2])  # permute n_steps and batch_size
        inputs = tf.reshape(inputs, [-1, 2*args.chunk_samples]) # (n_steps*batch_size, n_input)
        
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        inputs = tf.split(0, args.seq_length, inputs) # n_steps * (batch_size, n_hidden)
        
        # Get lstm cell output
        outputs, last_state = tf.nn.rnn(cell, inputs, initial_state=self.initial_state)

        #outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm_decode')
        output = tf.transpose(tf.pack(outputs), [1,0,2])
        output = tf.reshape(output, [-1, args.rnn_size])
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
        self.cost = lossfunc / (args.batch_size * args.seq_length * args.chunk_samples)
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

    def sample(self, sess, args, num=4410, start=None):

        def sample_gaussian(mu, sigma):
            return mu + (np.sqrt(sigma)*np.random.randn(*sigma.shape))

        if start is None:
            prev_x = np.random.randn(1, 1, 2*args.chunk_samples)
        elif len(start.shape) == 1:
            prev_x = start[np.newaxis,np.newaxis,:]
        prev_state = sess.run(self.cell.zero_state(1, tf.float32))

        chunks = np.zeros((num, 2*args.chunk_samples), dtype=np.float32)
        mus = np.zeros((num, args.chunk_samples), dtype=np.float32)
        sigmas = np.zeros((num, args.chunk_samples), dtype=np.float32)
        pis = np.zeros((num, args.num_mixture), dtype=np.float32)
       
        if len(start.shape) == 2:
            for i in range(start.shape[0]-1):
                prev_x = start[i,:]
                prev_x = prev_x[np.newaxis,np.newaxis,:]
                feed = {self.input_data: prev_x, self.initial_state:prev_state}
                [o_pi, o_mu, o_sigma, o_rho, prev_state] = sess.run([self.pi, self.mu, self.sigma, self.rho, self.final_state],feed)
            prev_x = start[-1,:]
            prev_x = prev_x[np.newaxis,np.newaxis,:]

        for i in xrange(num):
            feed = {self.input_data: prev_x, self.initial_state:prev_state}
            [o_pi, o_mu, o_sigma, o_rho, next_state] = sess.run([self.pi, self.mu, self.sigma, self.rho, self.final_state],feed)
            p = o_pi[0]
            #idx = np.argmax(p)
            if i%100 ==0:
                print np.argsort(p)
            if p.max() > 0.001:
                p[p<0.001] = 0.0
                p = p/p.sum()
            else:
                print p.max()
            p = (p-p.min())
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
