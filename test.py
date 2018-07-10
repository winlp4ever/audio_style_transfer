import tensorflow as tf
import numpy as np

import librosa
a = np.random.sample((1, 2, 3))
u = tf.Variable(a, dtype=tf.float32)
w = tf.transpose(u[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w).shape)