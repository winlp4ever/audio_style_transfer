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
from rainbowgram import plotcqtgram
from mdl import Cfg
from nsynth.wavenet import masked
from geter import decode

tf.logging.set_verbosity(tf.logging.INFO)

INS = ['bass', 'brass', 'flute', 'guitar',
       'keyboard', 'mallet', 'organ', 'reed',
       'string', 'synth_lead', 'vocal']


def mu_law_numpy(x, mu=255):
    out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    out = np.floor(out * 128)
    return out


class Net(object):
    def __init__(self, fpath, spath, tf_path, checkpoint_path, logdir, sr, length, batch_size=2 ** 12, stride=None):
        self.data = tf.data.TFRecordDataset([tf_path]).map(decode)
        self.checkpoint_path = checkpoint_path
        self.spath = spath
        self.logdir = logdir
        self.length = length
        self.sr = sr
        self.wavs, self.graph = self.build(fpath, length, sr)

    @staticmethod
    def build(fpath, length, sr, batch_size=2 ** 12, stride=None):
        if not stride:
            stride = batch_size // 2
        else:
            assert batch_size % stride == 0
        assert length % batch_size == 0
        wavs = []
        i = 0
        while i < length - batch_size + 1:
            wav, _ = librosa.load(fpath, sr=sr, mono=True)
            wav = wav[i:i + batch_size]
            wavs.append(np.reshape(wav, [1, -1]))
            i += stride
        wavs = np.concatenate(wavs, axis=0)

        print('\n wavs shape : {}\n'.format(wavs.shape))

        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=(np.zeros([1, batch_size])),
                trainable=True,
                name='regenerated_wav',
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

        return wavs, graph

    def knear(self, sess, type_s, type_t, k):
        it = self.data.make_one_shot_iterator()
        el = it.get_next()

        nb_frags = self.wavs.shape[0]

        N_s = [MyHeap(k) for _ in range(nb_frags)]
        N_t = [MyHeap(k) for _ in range(nb_frags)]

        encodings = np.concatenate([sess.run(self.graph['before_enc'], feed_dict={
            self.graph['quantized_input']: mu_law_numpy(self.wavs[i,])
        }) for i in range(nb_frags)], axis=1)

        i = 0
        try:
            while True:
                i += 1
                ins = sess.run(el['instrument_family'])

                if ins == type_s:
                    audio = np.reshape(sess.run(el['audio'][:self.length]), [1, self.length])
                    enc = sess.run(self.graph['before_enc'], feed_dict={
                        self.graph['quantized_input']: mu_law_numpy(audio)})

                    for j in range(nb_frags):
                        N_s[j].push((-norm(enc - encodings[j]), i, enc))

                    tf.logging.info(' sources - size {} - iterate {}'.format(len(N_s[0]), i))

                elif ins == type_t:
                    audio = np.reshape(sess.run(el['audio'][:self.length]), [1, self.length])
                    enc = sess.run(self.graph['before_enc'], feed_dict={
                        self.graph['quantized_input']: mu_law_numpy(audio)})

                    for j in range(nb_frags):
                        N_t[j].push((-norm(enc - encodings[j]), i, enc))

                    tf.logging.info(' targets - size {} - iterate {}'.format(len(N_t[0]), i))
        except tf.errors.OutOfRangeError:
            pass

        sources = np.concatenate([np.mean(heap.as_list(), axis=0) for heap in N_s], axis=0)
        targets = np.concatenate([np.mean(heap.as_list(), axis=0) for heap in N_t], axis=0)

        return encodings + sources - targets

    @staticmethod
    def get_encoding(encodings):
        print('\nshape encodings {} \n'.format(encodings.shape))
        nb_frags, batch_size, nb_channels = encodings.shape
        stride = batch_size // 2

        tf.logging.info('batch size: {}'.format(batch_size))
        l = stride * (nb_frags + 1)

        encoding = np.zeros([1, l, nb_channels], dtype=float)

        coefs_0 = np.array(
            [[[1 if j < stride else (batch_size - 1 - j) / (stride - 1) for _ in range(nb_channels)] for j in
              range(batch_size)]])
        coefs_1 = np.array([[[(stride - 1 - j) / (stride - 1) if j < stride else 1 for _ in range(nb_channels)] for j in
                             range(batch_size)]])
        coefs_2 = np.array(
            [[[(stride - 1 - j) / (stride - 1) if j < stride else (batch_size - 1 - j) / (stride - 1) for _ in
               range(nb_channels)] for j in range(batch_size)]])

        tf.logging.info('\n coefs shape: {}\n'.format(coefs_0.shape))

        encoding[:, 0: batch_size] = encodings[0] * coefs_0
        for i in range(1, nb_frags - 1):
            encoding[:, i * stride: i * stride + batch_size] += encodings[i] * coefs_2
        encoding[:,l - batch_size:] += encodings[nb_frags - 1] * coefs_1

        return encoding

    def test(self, sess):
        nb_frags = self.wavs.shape[0]

        encodings = np.concatenate([sess.run(self.graph['before_enc'], feed_dict={
            self.graph['quantized_input']: mu_law_numpy(self.wavs[i:i + 1, ])
        }) for i in range(nb_frags)], axis=0)

        encoding = self.get_encoding(encodings)
        return encoding


def main():
    fpath = './data/src/bass.wav'
    tf_path = './data/dataset/nsynth-valid.tfrecord'
    net = Net(fpath, None, tf_path, None, None, 16000, 16384)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        print(net.test(sess).shape)


if __name__ == '__main__':
    main()
