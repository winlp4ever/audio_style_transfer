import tensorflow as tf
import librosa
import numpy as np

fn = './data/tests/ep-44.wav'
aud, sr = librosa.load(fn, sr=16000)
aud = librosa.effects.pitch_shift(aud, sr=sr, n_steps=-4.0)
librosa.output.write_wav('./data/tests/m2f.wav', aud, sr=sr)