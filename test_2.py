import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [1, 2])
y = x**2
u = []
u.append(y)
y += 2
u.append(y)

with tf.Session() as sess:
    print(sess.run(u, feed_dict={x : np.array([[1, 2]])}))