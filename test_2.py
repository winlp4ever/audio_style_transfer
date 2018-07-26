import numpy as np
from scipy.linalg import ldl
from numpy.linalg import norm, eigh

import librosa


audio = np.zeros((16384 * 2,))
for i in range(2):
    aud, _ = librosa.load('./data/{}.wav'.format(i + 1), sr=16000)
    audio[i * 8192 * 2: (i+1) * 8192 * 2] = aud

librosa.output.write_wav('./data/pachel-crickets-64chnnls.wav', audio/np.max(audio), sr=16000)