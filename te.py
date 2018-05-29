import tensorflow as tf

u = tf.Variable(initial_value=0, dtype=tf.int32)

ass = tf.assign(u, u+1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ass.eval()
    print(type(u))
    print(sess.run(u))