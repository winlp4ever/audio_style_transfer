import tensorflow as tf
import numpy as np


with tf.Graph().as_default() as graph_1:
    x = tf.Variable(tf.random_normal(shape=(2,)), name='x')
    g = x ** 2

with tf.Session(graph=graph_1) as sess:
    print(sess.run(g, feed_dict={x : np.array([2, 1])}))
    print(sess.run(x))