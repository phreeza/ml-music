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

vae = VAE(z_dim=2,net_size=64)
ckpt = tf.train.get_checkpoint_state('save-vae')
vae.load_model('save-vae')
