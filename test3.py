import numpy as np
import librosa

fn = './data/src/crickets.wav'
aud, _ = librosa.load(fn, sr=16000, mono=False)
aud = aud[0, :16384 * 2]


librosa.output.write_wav('./data/src/crickets-2.wav', aud, sr=16000)