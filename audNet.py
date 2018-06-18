import tensorflow as tf
from geter import decode
import numpy as np
from myheap import MyHeap
from numpy.linalg import norm
import librosa
import os
import time
import argparse
from spectrogram import plotstft
from rainbowgram import plotcqt
from mdl import Cfg
from nsynth.wavenet import masked

tf.logging.set_verbosity(tf.logging.INFO)

INS = ['bass', 'brass', 'flute', 'guitar',
       'keyboard', 'mallet', 'organ', 'reed',
       'string', 'synth_lead', 'vocal']


def crt_fol(suppath, hour=False):
    dte = time.localtime()
    if hour:
        fol_n = os.path.join(suppath, '{}{}{}{}'.format(dte[1], dte[2], dte[3], dte[4]))
    else:
        fol_n = os.path.join(suppath, '{}{}'.format(dte[1], dte[2]))
    if not os.path.exists(fol_n):
        os.makedirs(fol_n)
    return fol_n


def get_s_name(s, t, fname, k, l, cmt):
    if s != t:
        return '{}2{}_{}_k{}_l{}{}_e'.format(INS[s], INS[t], fname, k, l, cmt)
    return '{}_k{}_l{}{}_e'.format(fname, k, l, cmt)


def mu_law_numpy(x, mu=255):
    out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    out = np.floor(out * 128)
    return out


def inv_mu_law_numpy(x, mu=255.0):
    x = np.array(x).astype(np.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu) ** np.abs(out) - 1)
    out = np.where(np.equal(x, 0), x, out)
    return out


class Net(object):
    def __init__(self, sourcepath, savepath, tf_path, checkpoint_path, logdir, sr, length):
        self.data = tf.data.TFRecordDataset([tf_path]).map(decode)
        self.checkpoint_path = checkpoint_path
        self.sourcepath = sourcepath
        self.savepath = savepath
        self.logdir = logdir
        self.length = length
        self.sr = sr
        self.graph = self.build(length)

    @staticmethod
    def build(length):
        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=(np.zeros([1, length])),
                trainable=True,
                name='regenerated_wav'
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

            comp = masked.pool1d(graph['before_enc'], 512, name='compress')

            graph.update({'comp': comp})

        return graph

    def load_model(self, sess):
        variables = tf.global_variables()
        variables.remove(self.graph['quantized_input'])

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.checkpoint_path)

    def get_embeds(self, sess, wavname, compress=False):
        wav, _ = librosa.load(os.path.join(self.sourcepath, wavname), sr=self.sr)
        wav = np.reshape(wav[:self.length], [1, self.length])
        if not compress:
            return sess.run(self.graph['before_enc'],
                       feed_dict={self.graph['quantized_input']: mu_law_numpy(wav)})

        return sess.run(self.graph['comp'],
                        feed_dict={self.graph['quantized_input']: mu_law_numpy(wav)})

    def bfgs(self, sess, encoding, epochs):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        with tf.name_scope('loss'):
            loss = tf.nn.l2_loss(self.graph['before_enc'] - encoding)
            tf.summary.scalar('loss', loss)

        summ = tf.summary.merge_all()

        i = 0

        def loss_tracking(loss_, summ_):
            nonlocal i
            tf.logging.info(' Step: {} -- Loss: {}'.format(i, loss_))
            writer.add_summary(summ_, global_step=i)
            i += 1

        with tf.name_scope('optim'):
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                var_list=[self.graph['quantized_input']],
                method='L-BFGS-B',
                options={'maxiter': 100})

        for ep in range(epochs):
            since = int(time.time())

            optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, summ])
            tf.logging.info(' Saving file ... Epoch: {} -- time-lapse: {}s'.format(ep, int(time.time() - since)))

            audio = sess.run(self.graph['quantized_input'])
            audio = inv_mu_law_numpy(audio)
            librosa.output.write_wav(self.savepath + '_' + str(ep) + '.wav', audio.T, sr=self.sr)

    def test(self, sess, input_name, target_name):
        input_path = os.path.join(self.sourcepath, input_name)
        target_path = os.path.join(self.sourcepath, target_name)

        sess.run(self.graph['comp'])

