from scipy.io import wavfile
from numpy.linalg import norm

import use
import numpy as np
import tensorflow as tf

x = tf.Variable(np.array([1, 2]), dtype=tf.float32)

m = tf.maximum(x, 1e-12) + tf.maximum(0.0, -x)
#y = x / m * tf.log(tf.real(m))
y = use.inv_mu_law(x)

loss = (y - 2) ** 2

optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                method='L-BFGS-B',
                options={'maxiter': 200})

def loss_tracking(x_value, l):
    print('value {} loss {}'.format(x_value, l))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[x, loss])


