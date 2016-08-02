import scipy
import numpy as np
import math
import numpy as np
import pydub


def MDCT(data, N, isInverse=False):
    """ Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """
    n_0 = (N/2. + 1.) / 2.
    n = np.linspace(0, N-1., N)

    out = np.array([])

    if isInverse:
        # Actually, extend K and the data
        K = np.linspace(0, N-1, N)
        mirror = -data[::-1]
        data = np.append(data, mirror)

        twiddle_exponents = np.multiply(K, 1j*2.*np.pi*n_0/N)
        twiddle = np.exp(twiddle_exponents)
        twiddle_signal = np.multiply(data, twiddle)

        ifft_data = np.fft.ifft(twiddle_signal)

        shifted_n = np.add(n, n_0)
        post_twiddle_exponents = np.multiply(shifted_n, 1j * np.pi / N)
        post_twiddle = np.exp(post_twiddle_exponents)

        out = np.multiply(ifft_data, post_twiddle)
        out = np.real(out)
        out = np.multiply(N, out)

    else:
        K = np.linspace(0, N/2. - 1, N/2)
        twiddle_exponents = np.multiply(n, -1j * np.pi / N)
        twiddle = np.exp(twiddle_exponents)
        twiddle_signal = np.multiply(data, twiddle)

        fft_data = np.fft.fft(twiddle_signal)
        
        shifted_K = np.add(K, 0.5)
        post_twiddle_exponents = np.multiply(shifted_K, -1j * (2*np.pi/N) * n_0)
        post_twiddle = np.exp(post_twiddle_exponents)

        out = np.multiply(2./N, fft_data[:N/2])
        out = np.multiply(post_twiddle, out)
        out = np.real(out)

    return out


def IMDCT(data, N):
    return MDCT(data, N, isInverse=True)


def slow_mdct(x,N):
    return np.array([np.sum([x[n]*cos(np.pi/N*(n+0.5*(N+1))*(k+0.5)) for n in range(2*N)]) for k in range(N)])
def slow_imdct(X,N):
    return (1./N)*np.array([np.sum([X[k]*cos(np.pi/N*(n+0.5*(N+1))*(k+0.5)) for k in range(N)]) for n in range(2*N)])


def chunk(data,N):
    assert len(data)%N == 0
    x = data.copy()
    x = x.reshape((-1,N))
    x = np.concatenate((x[:-1],x[1:]),axis=1)
    return x

def dechunk(chunks):
    x = np.zeros((chunks.shape[0]+1)*chunks.shape[1]/2)
    N = chunks.shape[1]/2
    for n in range(chunks.shape[0]):
        x[n*N:(n+2)*N]+=chunks[n,:]
    x[:N]+=chunks[0,:N]
    x[-N:]+=chunks[-1,-N:]
    return x


def loadf(fname):
    f = pydub.AudioSegment.from_mp3(fname)
    data = np.fromstring(f._data, np.int16)
    data = data.astype(np.float64).reshape((-1,2))
    data = data.mean(axis=1)
    data -= data.mean(axis=0)
    data /= data.std(axis=0) * 3.
    return data

def load_data(fname,N=1024):
    data = loadf(fname)
    
    data = data[:-(len(data)%N)]
    window = np.sin(np.pi/(2*N)*(np.arange(2*N)+0.5))
    c = chunk(data,N)
    spectrum = np.array([MDCT(window*cc,2*N) for cc in c])
     
    f = np.linspace(0,44100/2.,N)
    bark = 13*np.arctan(0.00076*f)+3.5*np.arctan((f/3500.)**2)
    bark_ind = bark.astype(int)
    energies = np.zeros((spectrum.shape[0],26))
    for n in range(26):
        energies[:,n] = np.sqrt(((spectrum[:,bark_ind==n]**2).sum(axis=1)))
    spectrum_norm = spectrum/energies[:,bark_ind]
    return np.hstack((np.log(energies),spectrum_norm))

def write_data(out,fname='out.wav'):
    from scipy.io import wavfile
    N = out.shape[1]-26
    window = np.sin(np.pi/(2*N)*(np.arange(2*N)+0.5))
    f = np.linspace(0,44100/2.,N)
    bark = 13*np.arctan(0.00076*f)+3.5*np.arctan((f/3500.)**2)
    bark_ind = bark.astype(int)

    energies_pred = np.exp(out[:,:26])
    print energies_pred[:10,:10]
    energies = np.zeros((out.shape[0],26))
    spectrum = out[:,26:]
    for n in range(26):
        energies[:,n] = np.sqrt(((spectrum[:,bark_ind==n]**2).sum(axis=1)))
    spectrum = spectrum/energies[:,bark_ind]*energies_pred[:,bark_ind]

    inv_chunks = np.array([window*IMDCT(ss,2*N) for ss in spectrum])
    inverse = dechunk(inv_chunks)
    wavfile.write(fname,44100,inverse/np.max(np.abs(inverse))) 
    return inverse
