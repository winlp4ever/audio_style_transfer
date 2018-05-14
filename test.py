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
        z = tf.Variable(tf.random_normal(shape=(1,), mean=-1, stddev=2.0))
        x = z
        g = x**2

    test= Test(2, graph=graph_1)
    with test.session as sess:
        sess.run(tf.variables_initializer([z]))
        print(type(x))
        print(sess.run(g))

