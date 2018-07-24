import numpy as np
from scipy.linalg import ldl
from numpy.linalg import norm, eigh

import librosa


audio = np.zeros((16384 * 2,))
for i in range(4):
    aud, _ = librosa.load('./data/{}.wav'.format(i + 1), sr=16000)
    audio[i * 8192: (i+1) * 8192] = aud

librosa.output.write_wav('./data/pachel-opera-24-500-100.wav', audio/np.max(audio), sr=16000)