import tensorflow as tf
import numpy as np
import time
from numpy.linalg import norm

from sklearn.decomposition import NMF

tf.logging.set_verbosity(tf.logging.INFO)

class MyNMF(object):
    def __init__(self, n_components, size_in, learning_rate=1e-4):
        assert len(size_in) == 2
        f, t = size_in

        with tf.device("/gpu:0"):
            tf.reset_default_graph()
            mat = tf.placeholder(tf.float32, shape=size_in, name='mat')
            W = tf.Variable(tf.random_uniform([f, n_components], minval=0., maxval=1.), dtype=tf.float32, name='W')
            H = tf.Variable(tf.random_uniform([n_components, t], minval=0., maxval=1.), dtype=tf.float32, name='H')

            clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
            clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
            clip = tf.group(clip_W, clip_H)

            with tf.name_scope('loss'):
                xent = tf.norm(mat - tf.matmul(W, H))
                tf.summary.scalar('nmf_loss', xent)


            with tf.name_scope('optim'):
                optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    xent,
                    method='L-BFGS-B',
                    options={'maxiter': 100})

            self.learning_rate = learning_rate
            self.size_in = size_in
            self.graph = {'mat_in' : mat,
                      'w' : W,
                      'h' : H,
                      'loss': xent,
                      'optim' : optimizer,
                      'clip': clip}

    def fit_transform(self, X, epochs):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:

            i = 0
            def loss_tracking(loss_):
                nonlocal i
                if not i % 5:
                    tf.logging.info(' Step: {} -- Loss: {}'.format(i, loss_))
                i += 1

            sess.run(tf.global_variables_initializer())

            for _ in range(epochs):
                self.graph['optim'].minimize(sess,
                                             loss_callback=loss_tracking,
                                             fetches=[self.graph['loss']],
                                             feed_dict={self.graph['mat_in']: X})
                sess.run(self.graph['clip'])

            tf.logging.info(' FINAL LOSS: {}'.format(sess.run(self.graph['loss'],
                                                              feed_dict={self.graph['mat_in']: X})))
            W, H = sess.run([self.graph['w'], self.graph['h']])
            print('my nmf error : {}'.format(norm(X - np.matmul(W, H))))
        return W / norm(W), H * norm(W)


def main():
    np.random.seed()
    n_components = 128
    size_in = [128, 320000]
    X = np.random.sample(size_in)
    nmf = MyNMF(n_components, size_in)
    since = time.time()
    W, H = nmf.fit_transform(X, epochs=10)

    assert (W >= 0).all() and (H >= 0).all()
    print('time-lapse {}'.format(time.time() - since))


    nmf_ = NMF(n_components=n_components, init='random', random_state=0, max_iter=400, solver='mu')
    since_ = time.time()
    W_ = nmf_.fit_transform(X)
    H_ = nmf_.components_
    print('time-lapse {}'.format(time.time() - since_))
    print('error: {}'.format(norm(X - np.matmul(W_, H_))))

if __name__ ==  '__main__':
    main()