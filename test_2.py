
import numpy as np

import scipy.io.wavfile as wav

rate, audio = wav.read('./test_data/sun.wav')
print(audio.shape)
audio = audio[4 * rate: 9 * rate,0]

print(rate)
wav.write('./test_data/gen_sun.wav', rate, audio)