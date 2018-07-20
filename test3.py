import numpy as np
import librosa

fn = './data/src/bass.wav'
aud, _ = librosa.load(fn, sr=16000)
audio = np.zeros((8192 * 10))
for i in range(10):
    audio[i * 8192: (i + 1) * 8192] = aud[: 8192]

librosa.output.write_wav('./data/src/bass_con.wav', audio, sr=16000)