import numpy as np
import tensorflow as tf

x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([[1], [2], [3]])
u = tf.data.Dataset.from_tensor_slices((x, y))
it = u.make_one_shot_iterator()
a, b= it.get_next()
c = b ** 2

with tf.Session() as sess:
    print(sess.run([a, b]))
    print(sess.run(c, feed_dict={b : np.array([2])}))
    print(sess.run([a, b]))
    print(sess.run([a, b]))