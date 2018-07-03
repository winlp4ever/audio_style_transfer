import tensorflow as tf
import numpy as np
import time
from sklearn.decomposition import NMF
from numpy.linalg import norm

tf.logging.set_verbosity(tf.logging.INFO)

EPSILON = np.finfo(np.float32).eps


def _multiplicative_update_w(X, W, H):
    num = tf.matmul(X, H, transpose_b=True)
    denum = tf.matmul(W, tf.matmul(H, H, transpose_b=True))
    denum = tf.maximum(tf.ones_like(denum) * EPSILON, denum)

    return num / denum


def _multiplicative_update_h(X, W, H):
    num = tf.matmul(W, X, transpose_a=True)
    denum = tf.matmul(W, tf.matmul(W, H), transpose_a=True)
    denum = tf.maximum(tf.ones_like(denum) * EPSILON, denum)
    # honorable mention : tf.fill()

    return num / denum


def mynmf(X, W=None, H=None, n_components=40, updt_w=True, updt_h=True, epochs=1000):
    print(' max {} -- min {}'.format(np.max(X), np.min(X)))
    assert (X >= 0).all()
    np.random.seed()
    f, t = X.shape
    avg = np.sqrt(X.mean() / n_components)

    if W is None:
        init_w = avg * np.random.randn(f, n_components)
        np.abs(init_w, init_w)
    else:
        init_w = W

    if H is None:
        init_h = avg * np.random.randn(n_components, t)
        np.abs(init_h, init_h)
    else:
        init_h = H
    with tf.device("/gpu:0"):
        x = tf.constant(X, dtype='float')
        w = tf.Variable(init_w, name='W', dtype='float')
        h = tf.Variable(init_h, name='H', dtype='float')

        loss = tf.norm(x - tf.matmul(w, h))

        update_w = w.assign(w * _multiplicative_update_w(x, w, h))
        update_h = h.assign(h * _multiplicative_update_h(x, w, h))

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())

        since = time.time()
        for i in range(epochs):
            if updt_w:
                sess.run(update_w)
            if updt_h:
                sess.run(update_h)
            c = sess.run(loss)
            if not i % 10:
                print('Epoch {0:} reached after {2:.2f}s -- loss {1:.4f}'.
                      format(i, c, time.time() - since), end='\r', flush=True)

        W, H =  sess.run([w, h])
        print('FINAL LOSS : {0:.4f}/{1:.4f} after {2:} epochs in {3:.4f}s'.
                        format(norm(X - np.matmul(W, H)), norm(X), epochs, time.time() - since))

    return W / norm(W), H * norm(W)


def main():
    n_components = 40
    epochs = 1000
    X = np.random.sample([128, 320000])
    w, h = mynmf(X, n_components=n_components, epochs=epochs)
    nmf_ = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000, solver='cd', verbose=1)
    since_ = time.time()
    W_ = nmf_.fit_transform(X)
    H_ = nmf_.components_
    print('time-lapse {}'.format(time.time() - since_))
    print('error: {}'.format(norm(X - np.matmul(W_, H_))))


if __name__ == '__main__':
    main()