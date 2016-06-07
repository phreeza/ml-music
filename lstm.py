import sys
sys.path.insert(0,'/home/mccolgan/local/lib/python2.7/site-packages/')
import numpy as np
from numpy import fft
import scipy
sys.path.append('/home/mccolgan/PyCharm Projects/keras')
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential
from keras.optimizers import RMSprop,Adagrad,SGD
from keras.layers.recurrent import LSTM
from theano import tensor as T

import pydub
import time


def stft(x, fftsize=1024, overlap=1):
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]  
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

def istft(X, overlap=1):   
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

f = pydub.AudioSegment.from_mp3('07_-_Brad_Sucks_-_Total_Breakdown.mp3')
data = np.fromstring(f._data, np.int16)
data = data.astype(np.float64).reshape((-1,2))
data -= data.min()
data /= data.max() / 2.
data -= 1.

data_stft = []
for n in range(2):
    data_stft.append(stft(data[:,n]))
data_stft = np.array(data_stft)
sadfadsf
freq = fft.fftfreq(1024)[:data_stft.shape[2]-1]
ph = freq[np.newaxis,:]*np.arange(data_stft.shape[1])[:,np.newaxis]%(2*np.pi)-np.pi

n_examples = 1024
n_frames_in = 100
nf = data_stft.shape[2]

X = np.zeros((n_examples*2,n_frames_in,data_stft.shape[0]*data_stft.shape[2]*2))
Y = np.zeros((n_examples*2,data_stft.shape[0]*nf*2))
for n in range(n_examples*2):
    i = np.random.randint(data_stft.shape[1]-n_frames_in-1)
    X[n,:,:nf] = np.log(np.abs(data_stft[0,i:i+n_frames_in,:]))
    X[n,:,nf:2*nf] = np.log(np.abs(data_stft[1,i:i+n_frames_in,:]))
    X[n,:,2*nf:3*nf] = np.angle(data_stft[0,i:i+n_frames_in,:])/(2*np.pi)
    X[n,:,3*nf:] = np.angle(data_stft[1,i:i+n_frames_in,:])/(2*np.pi)

    Y[n,:nf] = np.log(np.abs(data_stft[0,i+n_frames_in,:]))
    Y[n,nf:2*nf] = np.log(np.abs(data_stft[1,i+n_frames_in,:]))
    Y[n,2*nf:3*nf] = np.angle(data_stft[0,i+n_frames_in,:])/(2*np.pi)
    Y[n,3*nf:] = np.angle(data_stft[1,i+n_frames_in,:])/(2*np.pi)

scale = 10.*X[:,:,:2*nf].std()
X[:,:,:2*nf] = X[:,:,:2*nf]/scale
Y[:,:2*nf] = Y[:,:2*nf]/scale

#f = np.array([20.,25.,31.5.,40.,50.,63.,80.,100.,125.,160.,200.,
#              250.,315.,400.,500.,630.,800.,1000.,1250.,1600.,2000.,
#              2500.,3150.,4000.,5000.,6300.,8000.,10000.,12500.])
#af = np.array([0.532,0.506,0.480,0.455,0.432,0.409,0.387,0.367,0.349,0.330,0.315,
#                0.301,0.288,0.276,0.267,0.259,0.253,0.250,0.246,0.244,0.243,0.243,
#                0.243,0.242,0.242,0.245,0.254,0.271,0.301])

def phase_dist(y_true, y_pred):
    return (10.*T.sqrt(T.mean(T.square(y_pred[:,:2*nf] - y_true[:,:2*nf]), axis=-1))
            + T.mean(1. - T.cos(2*np.pi*(y_pred[:,2*nf:] - y_true[:,2*nf:])), axis=-1))

model = Sequential()
model.add(LSTM(1024,input_dim = X.shape[2], input_length = X.shape[1], return_sequences = True, activation = 'sigmoid'))
model.add(LSTM(1024, return_sequences = True, activation='sigmoid'))
model.add(LSTM(X.shape[2], return_sequences = False))
model.add(Flatten())
try:
    model.load_weights('lstm_3.h5')
except:
    pass
model.compile('rmsprop',phase_dist)
model.fit(X,Y,validation_split=0.5)
