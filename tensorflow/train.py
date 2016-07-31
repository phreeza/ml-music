import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from model import Model
import util

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rnn_size', type=int, default=256,
                     help='size of RNN hidden state')
  parser.add_argument('--num_layers', type=int, default=2,
                     help='number of layers in the RNN')
  parser.add_argument('--model', type=str, default='lstm',
                     help='rnn, gru, or lstm')
  parser.add_argument('--batch_size', type=int, default=50,
                     help='minibatch size')
  parser.add_argument('--seq_length', type=int, default=300,
                     help='RNN sequence length')
  parser.add_argument('--num_epochs', type=int, default=100,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=500,
                     help='save frequency')
  parser.add_argument('--grad_clip', type=float, default=10.,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate')
  parser.add_argument('--decay_rate', type=float, default=0.95,
                     help='decay rate for rmsprop')
  parser.add_argument('--num_mixture', type=int, default=20,
                     help='number of gaussian mixtures')
  parser.add_argument('--chunk_samples', type=int, default=1050,
                     help='number of samples per mdct chunk')
  parser.add_argument('--keep_prob', type=float, default=0.8,
                     help='dropout keep probability')
  args = parser.parse_args()
  train(args)

def next_batch(data, args):
    # returns a randomised, seq_length sized portion of the training data
    x_batch = []
    y_batch = []
    for i in xrange(args.batch_size):
        idx = np.random.randint(0, data.shape[0]-args.seq_length-2)
        x_batch.append(np.copy(data[idx:idx+args.seq_length]))
        y_batch.append(np.copy(data[idx+1:idx+args.seq_length+1]))
    return np.array(x_batch), np.array(y_batch)

def train(args):

    fname = '../Kimiko_Ishizaka_-_01_-_Aria.mp3'
    data = util.load_data(fname,args.chunk_samples-26)
    print data.shape
    with open(os.path.join('save', 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    model = Model(args)

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('logs', sess.graph)
        check = tf.add_check_numerics_ops()
        merged = tf.merge_all_summaries()
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        for e in xrange(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            state = model.initial_state.eval()
            for b in xrange(10):
                start = time.time()
                #t0 = np.random.randn(args.batch_size,1,(args.chunk_samples))
                #x = np.sin(2*np.pi*(np.arange(args.seq_length)[np.newaxis,:,np.newaxis]/30.+t0)) + np.random.randn(args.batch_size,args.seq_length,(args.chunk_samples))*0.1
                #y = np.sin(2*np.pi*(np.arange(1,args.seq_length+1)[np.newaxis,:,np.newaxis]/30.+t0)) + np.random.randn(args.batch_size,args.seq_length,(args.chunk_samples))*0.1
                x,y = next_batch(data,args)
                feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
                train_loss, state, _, cr, summary = sess.run([model.cost, model.final_state, model.train_op, check, merged], feed)
                summary_writer.add_summary(summary, e * 10 + b)
                end = time.time()
                print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * 10 + b,
                            args.num_epochs * 10,
                            e, train_loss, end - start)
                if (e * 10 + b) % args.save_every == 0 and ((e * 10 + b) > 0):
                    checkpoint_path = os.path.join('save', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * 10 + b)
                    print "model saved to {}".format(checkpoint_path)

if __name__ == '__main__':
  main()


