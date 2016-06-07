import sys
sys.path.insert(0,'/home/mccolgan/local/lib/python2.7/site-packages/')
import numpy as np
import theano
theano.config.floatX = 'float64'
from numpy import fft
import scipy
sys.path.append('/home/mccolgan/PyCharm Projects/keras')
from keras.layers.core import Flatten,Dense,Dropout
from keras.layers.convolutional import Convolution1D,Convolution2D
from keras.models import Graph,Sequential
from keras.optimizers import RMSprop,Adagrad,SGD
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.constraints import maxnorm
from theano import tensor as T

import pydub
import time

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and
            np.isnan(output[0]).any()):
            print('*** NaN detected ***')
            theano.printing.debugprint(node)
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            raise Exception

def stft(x, fftsize=512, overlap=1):
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]  
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

def istft(X, overlap=4):   
    fftsize=(X.shape[1]-1)*2
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop) 
    for n,i in enumerate(range(0, len(x)-fftsize, hop)): 
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x

#f = pydub.AudioSegment.from_mp3('07_-_Brad_Sucks_-_Total_Breakdown.mp3')

def loadf(fname):
    f = pydub.AudioSegment.from_mp3(fname)
    data = np.fromstring(f._data, np.int16)
    data = data.astype(np.float64).reshape((-1,2))
    data -= data.mean(axis=0)
    data /= data.std(axis=0) * 3.
    return data
#data = np.vstack([loadf('Kimiko_Ishizaka_-_01_-_Aria.mp3'),loadf('07_-_Brad_Sucks_-_Total_Breakdown.mp3')])
data = loadf('Kimiko_Ishizaka_-_01_-_Aria.mp3')
data_stft = []
for n in range(2):
    data_stft.append(stft(data[:,n]))
data_stft = np.array(data_stft)

n_frames_in = 22
n_examples = (data_stft.shape[1]-n_frames_in-1)
nf = data_stft.shape[2]

X = np.zeros((n_examples,4,nf+45,n_frames_in))
Y = np.zeros((n_examples,4,nf,1))

for n in range(n_examples):
    i = n#np.random.randint(data_stft.shape[1]-n_frames_in-1)
    X[n,0,22:22+nf,:] = np.angle(data_stft[0,i:i+n_frames_in,:nf]).T
    X[n,1,22:22+nf,:] = np.angle(data_stft[1,i:i+n_frames_in,:nf]).T
    X[n,2,22:22+nf,:] = np.log(np.abs(data_stft[0,i:i+n_frames_in,:nf])+1e-6).T
    X[n,3,22:22+nf,:] = np.log(np.abs(data_stft[1,i:i+n_frames_in,:nf])+1e-6).T

    Y[n,0,:,0] = np.angle(data_stft[0,i+n_frames_in,:nf])
    Y[n,1,:,0] = np.angle(data_stft[1,i+n_frames_in,:nf])
    Y[n,2,:,0] = np.log(np.abs(data_stft[0,i+n_frames_in,:nf])+1e-6)
    Y[n,3,:,0] = np.log(np.abs(data_stft[1,i+n_frames_in,:nf])+1e-6)

std_fact = 1.
Ym = Y.mean(axis=0)
Ym[:,:2] = 0. #Y.mean(axis=0)
Ys = Y.std(axis=0)*std_fact
Ys[:,:2] = 2*np.pi/np.sqrt(12.)*std_fact
Ys[Ys==0.] = 1.
Y = (Y-Ym)/Ys
def phase_dist(y_true, y_pred):
    return (std_fact*(T.mean(T.square(y_pred[:,2:,:,:] - y_true[:,2:,:,:]),axis=[1,2,3]))
            + T.mean((1. - T.cos(((y_pred*Ys+Ym)[:,:2,:,:] - (y_true*Ys+Ym)[:,:2,:,:]))),axis=[1,2,3]))/2

def phase_dist_split(y_true, y_pred):
    return std_fact*T.concatenate(
        [(1. - T.cos(((y_pred*Ys+Ym)[:,:,:2] - (y_true*Ys+Ym)[:,:,:2]))),
        T.square(y_pred[:,:,2:] - y_true[:,:,2:])],
        axis=2)

from scipy.io import wavfile
def reconstruct(Y_raw,name='foo'):
    Y = Y_raw*Ys+Ym
    sft = np.array([np.exp(1.0j * Y[:,2*nf:3*nf] + Y[:,:nf]),np.exp(1.0j * Y[:,3*nf:] + Y[:,nf:2*nf])])
    data_out = np.array([istft(sft[0,:,:]),istft(sft[1,:,:])])
    wavfile.write(name+'.wav',44100,(data_out).T)

model = Sequential()
model.add(Convolution2D(16,16,8,activation='tanh',border_mode='valid'  ,W_constraint=maxnorm(),b_constraint=maxnorm(),W_regularizer=l2(1e-5),input_shape=X.shape[1:],dim_ordering='th'))
model.add(Dropout(0.3))
model.add(Convolution2D(16,16,8,activation='tanh',border_mode='valid'  ,W_constraint=maxnorm(),b_constraint=maxnorm(),W_regularizer=l2(1e-5),dim_ordering='th'))
model.add(Dropout(0.3))
model.add(Convolution2D(4,16,8,activation='linear',border_mode='valid',W_constraint=maxnorm(),b_constraint=maxnorm(),W_regularizer=l2(1e-5),dim_ordering='th'))
#model.compile(SGD(clipnorm=0.1),loss=phase_dist,mode=theano.compile.MonitorMode(
#                        post_func=detect_nan))
model.compile(RMSprop(clipnorm=0.1),loss=phase_dist)
h = model.fit(X,Y,validation_split=0.2,batch_size=64,nb_epoch=1000)
