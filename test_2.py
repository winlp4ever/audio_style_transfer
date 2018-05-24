import tensorflow as tf
import numpy as np

import scipy.io.wavfile as wav

import heapq

u = []
heapq.heappush(u, (1, [[1], [2]]))
heapq.heappush(u, (2, [[2], [3]]))
heapq.heappush(u, (3, [[1], [4]]))

print(heapq.nlargest(2, u, key=lambda m : m[1]))

u = np.asarray(u)
print(u[:, 1])