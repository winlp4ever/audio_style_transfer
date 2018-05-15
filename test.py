import tensorflow as tf

class Test():
    def __init__(self, nb, graph):
        self.nb = nb
        self.session = tf.Session(graph=graph)


    def update(self, u):
        self.u = u

if __name__=='__main__':
    with tf.Graph().as_default() as graph_1:
        x = tf.placeholder(tf.float32, name='x1')
        y = tf.placeholder(tf.float32, name='y1')

        a = tf.Variable(tf.random_normal(shape=(1,), stddev=2.0), name='a')
        b = tf.Variable(tf.random_normal(shape=(1,)), name='b')
        g = a**2

    with tf.Session(graph=graph_1) as sess:

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


        saver = tf.train.Saver()
        saver.save(sess, save_path='./tmp/model')
        variables = tf.global_variables()
        v = [var for var in variables if var.op.name == 'a'][0]


    print(variables)
    variables.remove(v)
    print(variables)


    with tf.Session(graph=graph_1) as sess:
        x = tf.Variable(tf.random_normal(shape=(1,)), name='x')
        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, save_path='./tmp/model')
        sess.run(tf.variables_initializer([x]))
        print(sess.run(b))

