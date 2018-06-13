import librosa
from scipy.io import wavfile

fpath = './data/src/flute1.wav'

au, sr = librosa.load(fpath, sr=16000)

aus = []
for i in range(7):
    au_ = au[i * 2048: i * 2048 + 4096]
    wavfile.write('./data/src/flute1{}.wav'.format(i), sr, au_)
