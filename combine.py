import numpy as np
import librosa
from scipy.io import wavfile

batch_size = 4096
stride = batch_size // 2


coef0 = np.array([1 if i < stride else (batch_size - i - 1) / (stride - 1) for i in range(batch_size)])
coef1 = np.array([i / (stride - 1) if i < stride else 1 for i in range(batch_size)])
coef2 = np.array([i / (stride - 1) if i < stride else (batch_size - i - 1) / (stride - 1) for i in range(batch_size)])

au = np.zeros(shape=(16384,), dtype=float)

for i in range(7):
    au_, _ = librosa.load('./data/test/flute{}.wav'.format(i), sr = 16000)
    if i == 0:
        coef = coef0
    elif i == 6:
        coef = coef1
    else:
        coef = coef2
    au[i * stride: i * stride + batch_size] += au_ * coef
wavfile.write('./data/test/flute.wav', 16000, au)
