import tensorflow as tf
import numpy as np

import librosa

mu = 255
a = np.random.sample((1, 2, 3))
u = tf.Variable(a, dtype=tf.float32)
w = tf.maximum(a)
w = tf.maximum(w, 1e-5)
#w = tf.nn.l2_loss(u)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))