import numpy as np
from numpy.linalg import norm
import time
from sklearn.decomposition import NMF

import tensorflow as tf

u = tf.constant(3.)

with tf.Session() as sess:
    with tf.Session() as sess_:
        print(sess.run(u))