from scipy.io import wavfile
from numpy.linalg import norm

import use
import numpy as np
import tensorflow as tf

c = []
with tf.device('/gpu:0'):
    x = tf.Variable(np.array([1, 2]), dtype=tf.float32)
    m = tf.maximum(x, 1e-12) + tf.maximum(0.0, -x)
    c.append(x)
with tf.device('/gpu:1'):
    z = tf.Variable(np.array([3, 1]), dtype=tf.float32)
    a = tf.constant([1.0, 2.0])
    c.append(a)
    c.append(z)
#y = x / m * tf.log(tf.real(m))
y = use.inv_mu_law(x)


loss = (y - 2) ** 2 + (z - 3) ** 2
c.append(loss)
sum = tf.add_n(c)

optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                method='L-BFGS-B',
                options={'maxiter': 200})

def loss_tracking(x_value, l):
    print('value {} loss {}'.format(x_value, l))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(sum))
    optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[x, loss])


