import tensorflow as tf
import numpy as np

import librosa

fn = './data/ep-test-89.wav'

aud, _ = librosa.load(fn, sr=16000)

aud = aud/np.mean(aud)
librosa.output.write_wav('./data/test.wav', aud, sr=16000)