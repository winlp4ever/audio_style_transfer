import librosa
from scipy.io import wavfile
import numpy as np
fpath = './data/src/flute1.wav'

au0, sr = librosa.load('./data/test/0.wav', sr=16000)
au1, _ = librosa.load('./data/test/1.wav', sr=16000)
au2, _ = librosa.load('./data/test/2.wav', sr=16000)
au3, _ = librosa.load('./data/test/3.wav', sr=16000)

au = np.zeros((4096*4,), dtype=float)
au[:4096] = au0
au[4096:4096*2]=au1
au[4096*2:4096*3]=au2
au[4096*3:]=au3

wavfile.write('./data/test/flute.wav', sr, au)
