import numpy as np
import librosa

fn = './data/src/canon.wav'
aud, _ = librosa.load(fn, sr=16000, mono=False)
aud = aud[0, 60 * 16000: ]


librosa.output.write_wav('./data/src/canon_mono.wav', aud, sr=16000)