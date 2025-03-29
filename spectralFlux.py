"""
SpectralFlux.py
Compute the spectral flux between consecutive spectra
This technique can be for onset detection

rectify - only return positive values
"""
import numpy as np
from scipy import signal


def spectralFlux(x, fs, rectify=False):
    """
    Compute the spectral flux between consecutive spectra
    """
    f, t, spectra = signal.spectrogram(x, fs)
    spectralFlux = [np.sum(spectra[0])]

    # Compute flux for subsequent spectra
    for s in range(1, len(spectra)):
        prevSpectrum = spectra[s - 1]
        spectrum = spectra[s]

        Diff = np.abs(spectrum) - np.abs(prevSpectrum)
        if rectify:
            Diff[Diff < 0] = 0
        flux = np.sum(Diff)
        spectralFlux.append(flux)
    return spectralFlux


if __name__ == '__main__':
    spectralFlux = spectralFlux(np.random.random(100000), 4000)
    breakpointx = 546
