import numpy as np
import tensorflow as tf
a = np.ones([1, 64, 4])
b = np.split(a, 4, axis=1)
c = np.mean(b, axis=0)
print(c.shape)
