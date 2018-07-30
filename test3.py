import librosa
import numpy as np

aud, sr = librosa.load('./data/ep-9.wav', sr=16000)
aud = aud[100:]
librosa.output.write_wav('./data/ep.wav', aud/np.max(aud), sr=16000)