import tensorflow as tf
import numpy as np
from mod import Cfg
import librosa

'''
mu = 255
a = np.random.sample(())
u = tf.Variable(a, dtype=tf.float32)
w = tf.reduce_max(a)
w = tf.maximum(w, 1e-5)
v = tf.sign(u) * u
x_quantized = tf.sign(u) * tf.log(1 + mu * tf.abs(u)) / np.log(1 + mu)
x_scaled = x_quantized
x_scaled = tf.expand_dims(x_scaled, 0)
#w = tf.nn.l2_loss(u)
loss = (x_scaled - 4) ** 2
optim = tf.train.AdamOptimizer()
optimizer = optim.minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(optimizer)
        if not i % 100:
            val, l = sess.run([u, loss])
            print('ep {} val \n{} loss {}'.format(i, val, l))
'''

config = Cfg()
tf.reset_default_graph()
with tf.device("/gpu:0"):
    x = tf.Variable(
        initial_value=(np.ones([1, 16384])),
        trainable=True,
        name='regenerated_wav',
        dtype=tf.float32
    )

    graph = config.build({'wav': x}, is_training=True)
    graph.update({'input': x})

    stl = []
    for i in range(20):
        embeds = tf.stack([config.extracts[j][0, :, i] for j in range(10, 20)], axis=1)
        embeds = tf.matmul(embeds, embeds, transpose_a=True)
        m = tf.reduce_max(embeds)
        m = tf.maximum(m, 1e-10)
        stl.append(embeds / m)

    style_embeds = tf.stack(stl, axis=0)

loss = tf.nn.l2_loss(style_embeds - 12)
optim = tf.train.AdamOptimizer(1e-5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, l = sess.run([optim, loss])
        print('ep {} loss {}'.format(i, l))
