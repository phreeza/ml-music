# IPython log file
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from modelVAE import VAE

from matplotlib import pyplot as plt

vae = VAE(z_dim=64,net_size=128,chunk_samples=128)
ckpt = tf.train.get_checkpoint_state('save-vae-aws')
vae.load_model('save-vae-aws')
