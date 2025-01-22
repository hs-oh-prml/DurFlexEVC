import numpy as np
from scipy.io import wavfile


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav = wav * 32767
    wavfile.write(path[:-4] + ".wav", sr, wav.astype(np.int16))
