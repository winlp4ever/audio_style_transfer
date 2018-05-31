import tensorflow as tf
import numpy as np

import scipy.io.wavfile as wav

<<<<<<< HEAD
rate, audio = wav.read('./test_data/pia.wav')
print(audio.shape)
print(len(audio[:,0]))
pia = np.zeros(shape=(rate*4,))
print(pia.shape)
pia[: len(audio[:,0])] = audio[:,0]
print(pia[:len(audio[:,0])]-audio[:,0])

print(rate)
wav.write('./test_data/gen_pia.wav', rate, pia)
=======
import heapq

u = []
heapq.heappush(u, (1, [[1], [2]]))
heapq.heappush(u, (2, [[2], [3]]))
heapq.heappush(u, (3, [[1], [4]]))

print(heapq.nlargest(2, u, key=lambda m : m[1]))

u = np.asarray(u)
print(u[:, 1])
>>>>>>> d30befe2e197253d457ed4424ad8b4aa951459fe
