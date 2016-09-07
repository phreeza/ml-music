# IPython log file
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle
import util

from modelVAE import VAE

from matplotlib import pyplot as plt

data , means, stds = util.load_augment_data(util.loadf('../mp3/Kimiko Ishizaka - J.S. Bach- -Open- Goldberg Variations, BWV 988 (Piano) - 01 Aria.mp3'),1024)

vae = VAE(z_dim=256,net_size=512,chunk_samples=1024)
ckpt = tf.train.get_checkpoint_state('save-vae')
vae.load_model('save-vae')

x = np.zeros((2000,1024))
vz = np.random.randn(1,256)
z = np.random.randn(1,256)
zh = []
for n in range(2000):
    z += 0.08*(-0.5*z + 3*np.random.randn(*z.shape))
    zh.append(np.sqrt(np.sum(z**2)))
    mu,s = vae.generate(z)
    x[n,:] = (mu+0.00*np.sqrt(np.exp(s))*np.random.randn(*mu.shape)).squeeze()

out = np.zeros((2*1024,2000))
out[:1024,:] = (x*stds+means).T
out[1024:,:] = 1-2*np.random.randint(2,size=(1024,2000))
sample_trace = util.write_data(np.minimum(out.T,1.1), fname = "out-vae.wav")
