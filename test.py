import numpy as np
from numpy.linalg import norm
import time
from sklearn.decomposition import NMF

import tensorflow as tf

u = tf.constant(np.array([-1, 2]))
v = tf.abs(u)
with tf.Session() as sess:
    print(sess.run(v))