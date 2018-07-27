import numpy as np
from scipy.linalg import ldl
from numpy.linalg import norm, eigh

import librosa


fn = './data/pachel-drums.wav'
aud, sr = librosa.load(fn, sr=16000)
librosa.output.write_wav(fn, aud / np.mean(aud), sr=16000)