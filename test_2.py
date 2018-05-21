
import numpy as np

import scipy.io.wavfile as wav

rate, audio = wav.read('./test_data/pia.wav')
print(audio.shape)
print(len(audio[:,0]))
pia = np.zeros(shape=(rate*4,))
print(pia.shape)
pia[: len(audio[:,0])] = audio[:,0]
print(pia[:len(audio[:,0])]-audio[:,0])

print(rate)
wav.write('./test_data/gen_pia.wav', rate, pia)