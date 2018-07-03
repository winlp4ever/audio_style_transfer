import tensorflow as tf
import numpy as np

import librosa

fpath = './data/src/exo_flute.wav'
out = './data/src/exo_flute.wav'
aud, sr = librosa.load(fpath, sr=16000)
#aud = aud[4 * sr:]

librosa.output.write_wav(out, aud, sr)

