import tensorflow as tf

import os
import cPickle
import util
from model import Model
import numpy as np

with open(os.path.join('save', 'config.pkl')) as f:
    saved_args = cPickle.load(f)

data , means, stds = util.load_augment_data(util.loadf('../mp3/Kimiko Ishizaka - J.S. Bach- -Open- Goldberg Variations, BWV 988 (Piano) - 01 Aria.mp3'),saved_args.chunk_samples)

model = Model(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state('save')
print "loading model: ",ckpt.model_checkpoint_path

saver.restore(sess, ckpt.model_checkpoint_path)
n = np.random.randint(data.shape[0]-100)
sample_data,mus,sigmas,pis = model.sample(sess,saved_args,start=data[n:n+100,:])
sample_data[:,:saved_args.chunk_samples] = sample_data[:,:saved_args.chunk_samples]*stds + means
data[:,:saved_args.chunk_samples] = data[:,:saved_args.chunk_samples]*stds + means
sample_trace = util.write_data(np.minimum(sample_data,1.1), fname = "out.wav")
util.write_data(data[500:1700,:], fname = "out_ref.wav")
