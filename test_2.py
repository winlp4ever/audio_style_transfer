import tensorflow as tf
import numpy as np


with tf.Graph().as_default() as graph_1:
    x = tf.Variable(tf.random_normal(shape=(2,)), name='x')
    g = x ** 2
    z = g
    u = g + x

with tf.Session(graph=graph_1) as sess:
    print(sess.run(u, feed_dict={z : np.array([2, 1]), x : np.array([3, 2])}))
