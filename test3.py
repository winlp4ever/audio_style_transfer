import tensorflow as tf
import numpy as np
a = tf.constant(np.random.sample([2, 3, 4]))
b = tf.constant(np.random.sample([2, 3, 4]))
c = tf.transpose(b, perm=[0, 2, 1])
d = tf.matmul(a, c)
e = tf.nn.l2_normalize(d, axis=(1, 2))
with tf.Session() as sess:
    print(sess.run(d))
    print(sess.run(tf.shape(d)))
    print(sess.run(e))
    print(sess.run(tf.shape(e)))
    print(sess.run([tf.norm(e[i]) for i in range(2)]))