import tensorflow as tf

step = tf.Variable(1, dtype=tf.float32)
ass = tf.assign(step, step + 1)

vector = tf.Variable([7., 7.], 'vector')
loss = tf.nn.l2_loss(tf.square(vector))
tf.summary.scalar('loss', loss)

summ = tf.summary.merge_all()
optimizer = tf.contrib.opt.ScipyOptimizerInterface(
    loss, method='L-BFGS-B',
    options={'maxiter': 100})

writer = tf.summary.FileWriter('./log')
with tf.Session() as sess:
    i = 0
    def print_loss(loss_evaled, vector_evaled, summ_):
        global i
        print(loss_evaled, vector_evaled)
        writer.add_summary(summ_, global_step=i)
        i += 1

    tf.global_variables_initializer().run()
    writer.add_graph(sess.graph)
    optimizer.minimize(sess,
                       loss_callback=print_loss,
                       fetches=[loss, vector, summ])
    print(vector.eval())
    for i in range(5):
        sess.run(ass)
        print(sess.run(step))
