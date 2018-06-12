import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

f1 = './data/src/bass.wav'
f2 = './data/src/voice.wav'

import librosa

au1, sr = librosa.load(f1, sr=16000)
au2, _ = librosa.load(f2, sr=16000)

ps, mags = librosa.core.piptrack(au1[:16000], sr)
print(ps.shape)