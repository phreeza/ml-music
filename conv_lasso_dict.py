import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d


class ConvDictLearn(object):
    def __init__(self, n_tracks, batch_size, track_len, n_filters, filter_len, alpha, D_init=None):
        self.A = theano.shared(
            value=1e-5 * numpy.random.randn(
                n_tracks, n_filters, track_len - filter_len + 1).astype(
                theano.config.floatX
            ),
            name='A',
            borrow=True
        )
        if D_init is None:
            D_init = numpy.random.randn(
                n_filters, filter_len)
        D_init = D_init / numpy.sqrt((D_init ** 2).sum(axis=1).reshape(D_init.shape[0], 1))
        self.D = theano.shared(
            value=D_init.astype(
                theano.config.floatX
            ),
            name='D',
            borrow=True
        )

        self.alpha = alpha
        self.n_tracks = n_tracks
        self.batch_size = batch_size

        # parameters of the model
        self.params = [self.A, self.D]

    def prediction_A(self):
        def norm(x):
            return x / T.sqrt((x * x).sum(axis=1, keepdims=True))

        return conv2d(self.A.dimshuffle(0, 1, 'x', 2), norm(self.D).dimshuffle('x', 0, 'x', 1), border_mode='full')[:,
               0, 0, :]

    def cost_A(self, X):
        return T.std((X - self.prediction_A())) + self.alpha * T.mean(abs(self.A))

    def prediction_D(self, index):
        def norm(x):
            return x / T.sqrt((x * x).sum(axis=1, keepdims=True))

        return conv2d(self.A[index * self.batch_size: (index + 1) * self.batch_size,:,:].dimshuffle(0, 1, 'x', 2),
                      norm(self.D).dimshuffle('x', 0, 'x', 1), border_mode='full')[:, 0, 0, :]

    def cost_D(self, X, index):
        return T.std(X - self.prediction_D(index))


def sgd_optimization_dict(learning_rate_A=0.001, learning_rate_D=0.1, n_epochs_outer=10, n_epochs_A=30, n_epochs_D=30,
                          batch_size=4):
    import pydub
    import time
    import numpy as np

    f = pydub.AudioSegment.from_mp3('07_-_Brad_Sucks_-_Total_Breakdown.mp3')
    data = np.fromstring(f._data, np.int16)
    data = data[::2].astype(np.float64) + data[1::2].astype(np.float64)
    data -= data.min()
    data /= data.max() / 2.
    data -= 1.

    track_seconds = 4.
    track_len = int(44100 * track_seconds)
    n_tracks = min(1000, (data.shape[0] // track_len))

    D_init = np.zeros((100, 44100 / 400))
    for n in range(100):
        index = np.random.randint(0, len(data) - 44100 / 400)
        D_init[n, :] = data[index:index + 44100 / 400]
    data = data[:track_len * n_tracks].reshape((-1, track_len))
    data = theano.shared(data.astype(theano.config.floatX))

    n_batches = data.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()

    X = T.matrix('X')
    learner = ConvDictLearn(n_tracks, batch_size, track_len, n_filters=100, filter_len=44100 / 400, alpha=1e4, D_init=D_init)

    cost_A = learner.cost_A(X)
    cost_D = learner.cost_D(X)

    g_A = T.grad(cost=cost_A, wrt=learner.A)
    g_D = T.grad(cost=cost_D, wrt=learner.D)

    updates_A = [(learner.A, learner.A - learning_rate_A * g_A)]
    updates_D = [(learner.D, learner.D - learning_rate_D * g_D)]

    train_A = theano.function(
        inputs=[],
        outputs=cost_A,
        updates=updates_A,
        givens={
            X: data
        }
    )

    train_D = theano.function(
        inputs=[index],
        outputs=cost_D,
        updates=updates_D,
        givens={
            X: data[index * batch_size: (index + 1) * batch_size, :],
            index: index
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    start_time = time.clock()

    for e_outer in xrange(n_epochs_outer):
            for e_A in xrange(n_epochs_A):
                cost = train_A()
                print(
                    'epoch %i, A-step %i cost %f' %
                    (
                        e_outer,
                        e_A,
                        cost
                    )
                )

            for e_D in xrange(n_epochs_D):
                cost = train_D(minibatch_index)
                for minibatch_index in xrange(n_batches):
                    print(
                        'epoch %i, minibatch %i/%i, D-step %i cost %f' %
                        (
                            e_outer,
                            minibatch_index + 1,
                            n_batches,
                            e_D,
                            cost
                        )
                    )

    end_time = time.clock()
    print "Running time", end_time - start_time
    return learner


if __name__ == '__main__':
    learner = sgd_optimization_dict()
