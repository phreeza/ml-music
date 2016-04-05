import sys
sys.path.insert(0,'/home/mccolgan/local/lib/python2.7/site-packages/')
import numpy as np
sys.path.append('/home/mccolgan/PyCharm Projects/keras')
from keras.layers.core import Layer
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential
from theano import tensor as T

import pydub
import time
class Unpooling1D(Layer):
    def __init__(self, subsample_length=2):
        super(Unpooling1D,self).__init__()
        self.input = T.tensor3()
        self.subsample_length = subsample_length

    def get_output(self, train):
        X = self.get_input(train)
        output = X.repeat(self.subsample_length, axis=1)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "subsample_length":self.subsample_length}

filter_len = 44100
subsample = 4410
filters = 200
model = Sequential()
model.add(Convolution1D(2,filters,filter_len,subsample_length=subsample,activation='tanh'))
model.add(Unpooling1D(subsample_length=subsample))
model.add(Convolution1D(filters,2,filter_len, border_mode='full',activation='tanh'))
model.compile('sgd','mse')

f = pydub.AudioSegment.from_mp3('07_-_Brad_Sucks_-_Total_Breakdown.mp3')
data = np.fromstring(f._data, np.int16)[:filter_len*10]
data = data.astype(np.float64).reshape((1,-1,2))
data = data[:,:subsample*int(len(data)/subsample)-1,:]
data -= data.min()
data /= data.max() / 2.
data -= 1.

#print np.corrcoef(data[0,:,0],data[0,:,1])
#from keras.utils.layer_utils import print_layer_shapes
#print_layer_shapes(model,data.shape)

model.fit(data,data)
