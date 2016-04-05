import sys
sys.path.insert(0,'/home/mccolgan/local/lib/python2.7/site-packages/')
import numpy as np
import theano
theano.config.floatX = 'float64'
from numpy import fft
import scipy
sys.path.append('/home/mccolgan/PyCharm Projects/keras')
from keras.layers.core import Flatten,Dense,Dropout
from keras.layers.convolutional import Convolution1D
from keras.models import Graph,Sequential
from keras.optimizers import RMSprop,Adagrad,SGD
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from theano import tensor as T

import pydub
import time


def stft(x, fftsize=1024, overlap=1):
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
print data.shape
data_stft = []
for n in range(2):
    data_stft.append(stft(data[:,n]))
data_stft = np.array(data_stft)

n_frames_in = 6
n_examples = (data_stft.shape[1]-n_frames_in-1)
nf = data_stft.shape[2]

X = np.zeros((n_examples,n_frames_in,data_stft.shape[0]*nf*2))
Y = np.zeros((n_examples,data_stft.shape[0]*nf*2))

for n in range(n_examples):
    i = n#np.random.randint(data_stft.shape[1]-n_frames_in-1)
    X[n,:,:nf] = np.log(np.abs(data_stft[0,i:i+n_frames_in,:nf])+1e-6)
    X[n,:,nf:2*nf] = np.log(np.abs(data_stft[1,i:i+n_frames_in,:nf])+1e-6)
    X[n,:,2*nf:3*nf] = np.angle(data_stft[0,i:i+n_frames_in,:nf])
    X[n,:,3*nf:] = np.angle(data_stft[1,i:i+n_frames_in,:nf])

    Y[n,:nf] = np.log(np.abs(data_stft[0,i+n_frames_in,:nf])+1e-6)
    Y[n,nf:2*nf] = np.log(np.abs(data_stft[1,i+n_frames_in,:nf])+1e-6)
    Y[n,2*nf:3*nf] = np.angle(data_stft[0,i+n_frames_in,:nf])
    Y[n,3*nf:] = np.angle(data_stft[1,i+n_frames_in,:nf])

std_fact = 10.
Ym = Y.mean(axis=0)
Ys = Y.std(axis=0)*std_fact
Ys[Ys==0.] = 1.
Y = (Y-Ym)/Ys
def phase_dist(y_true, y_pred):
    
    #return (T.mean((1. - T.cos(((y_pred*Ys+Ym)[:,2*nf:] - (y_true*Ys+Ym)[:,2*nf:]))), axis=-1)
    #        + 10*T.mean(T.maximum(T.abs_((y_true - y_pred)*Ys)[:,2*nf:]-3*np.pi,0.),axis=-1))
    return (std_fact*T.sqrt(T.mean(T.square(y_pred[:,:2*nf] - y_true[:,:2*nf]), axis=-1))
            + T.mean(T.exp((y_true*Ys+Ym)[:,:2*nf])*(1. - T.cos(((y_pred*Ys+Ym)[:,2*nf:] - (y_true*Ys+Ym)[:,2*nf:]))), axis=-1)/T.mean(T.exp((y_true*Ys+Ym)[:,:2*nf]))
            + 10*T.mean(T.maximum(T.abs_((y_true - y_pred)*Ys)[:,2*nf:]-3*np.pi,0.),axis=-1))

def phase_dist_split(y_true, y_pred):
    return (std_fact*T.sqrt(T.mean(T.square(y_pred[:,:2*nf] - y_true[:,:2*nf]), axis=-1)),
            2*T.mean(T.exp((y_true*Ys+Ym)[:,:2*nf])*(1. - T.cos(((y_pred*Ys+Ym)[:,2*nf:] - (y_true*Ys+Ym)[:,2*nf:]))), axis=-1) , 10*T.mean(T.maximum(T.abs_((y_true - y_pred)*Ys)[:,2*nf:]-3*np.pi,0.),axis=-1))

from scipy.io import wavfile
def reconstruct(Y_raw,name='foo'):
    Y = Y_raw*Ys+Ym
    sft = np.array([np.exp(1.0j * Y[:,2*nf:3*nf] + Y[:,:nf]),np.exp(1.0j * Y[:,3*nf:] + Y[:,nf:2*nf])])
    data_out = np.array([istft(sft[0,:,:]),istft(sft[1,:,:])])
    wavfile.write(name+'.wav',44100,(data_out).T)

model = Sequential()
model.add(Flatten(input_shape=(X.shape[1],X.shape[2])))
model.add(Dense(8*nf,activation='relu',W_regularizer=l2(1e-6), input_shape=(Y.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(5*nf,activation='relu',W_regularizer=l2(1e-6)))
model.add(Dropout(0.3))
model.add(Dense(Y.shape[1],activation='tanh'))

model.compile(RMSprop(clipnorm=0.1),loss=phase_dist)
h = model.fit(X,Y,validation_split=0.2,batch_size=128,nb_epoch=1000)
