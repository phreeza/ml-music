f = np.linspace(0,44100/2.,1024)
bark = 13*arctan(0.00076*f)+3.5*arctan((f/3500.)**2)
bark_ind = bark.astype(int)
energies = np.zeros((spectrum.shape[0],26))
for n in range(26):
    energies[:,n] = np.sqrt(((spectrum[:,bark_ind==n]**2).sum(axis=1)))
spectrum_norm = spectrum/energies[:,bark_ind]
