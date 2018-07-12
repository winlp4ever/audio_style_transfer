import tensorflow as tf
import matplotlib.pyplot as plt
from mdl import Cfg
import numpy as np
import use
import librosa
import argparse
import os

plt.switch_backend('agg')

def build_graph(length, layer_ids):
    config = Cfg()
    with tf.device("/gpu:0"):
        x = tf.Variable(
            initial_value=(np.zeros([1, length])),
            trainable=True,
            name='regenerated_wav'
        )

        graph = config.build({'quantized_wav': x}, is_training=True)

        embeds = tf.concat([config.extracts[i] for i in layer_ids], axis=2)[0][:, ::128]
        graph.update({'embeds': embeds})

    return graph

def load_model(graph, sess, checkpoint_path):
    vars = tf.global_variables()
    vars.remove(graph['quantized_input'])

    saver = tf.train.Saver(vars)
    saver.restore(sess, checkpoint_path)

def get_embeds(graph, sess, aud):
    if len(aud.shape) == 1:
        aud = np.reshape(aud, [1, -1])
    return sess.run(graph['embeds'], feed_dict={graph['quantized_input']: use.mu_law_numpy(aud)})

def read_file(filename, length, sr=16000):
    aud, _ = librosa.load(filename, sr=sr)
    auds = [aud[i * length: (i + 1) * length] for i in range(len(aud) // length)]
    return auds

def compare2embeds(embeds1, embeds2, layer_ids, figdir=None, ord=None):
    embeds1 = np.log(embeds1 + 1)
    embeds2 = np.log(embeds2 + 1)
    nb_lyrs = len(layer_ids)
    fig, axs = plt.subplots(1, 2, figsize=(30, 10))
    ptp = np.dot(embeds1.T, embeds1)
    ete = np.dot(embeds2.T, embeds2)
    axs[0].imshow(ptp, interpolation='nearest', cmap=plt.cm.plasma)
    axs[1].imshow(ete, interpolation='nearest', cmap=plt.cm.plasma)
    axs[0].set_title('{}'.format(layer_ids))
    assert figdir, 'figdir must not be None'
    assert ord is not None
    plt.savefig(os.path.join(figdir, 'p{}'.format(ord)))


def get_path(figdir, fn1, fn2, layers):
    path = use.crt_t_fol(figdir)
    s = ''
    for i in layers:
        s += '_' + str(i)
    path = os.path.join(path, 'compare::{}-{}{}'.format(fn1, fn2, s))
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class ShowNet(object):
    def __init__(self, srcdir, ckpt_path, figdir, layer_ids, length=16384, sr=16000):
        self.graph = build_graph(length, layer_ids)
        self.srcdir = srcdir
        assert ckpt_path, 'must provide a ckpt path for this model!'
        self.ckpt_path = ckpt_path
        self.figdir = figdir
        self.sr = sr
        self.length = length
        self.layer_ids = layer_ids

    def compare(self, fn1, fn2):
        fp1 = os.path.join(self.srcdir, fn1 + '.wav')
        fp2 = os.path.join(self.srcdir, fn2 + '.wav')

        auds1 = read_file(fp1, self.length)
        auds2 = read_file(fp2, self.length)
        figdir = get_path(self.figdir, fn1, fn2, self.layer_ids)

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            load_model(self.graph, sess, self.ckpt_path)

            embeds1 = [get_embeds(self.graph, sess, aud) for aud in auds1]
            embeds2 = [get_embeds(self.graph, sess, aud) for aud in auds2]

            for i in range(min(len(embeds1), len(embeds2))):
                compare2embeds(embeds1[i], embeds2[i], self.layer_ids, figdir, i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    parser.add_argument('--srcdir', nargs='?', default='./data/src')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--layers', nargs='*', default=[i for i in range(10,20)], type=int)
    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')

    args = parser.parse_args()

    net = ShowNet(args.srcdir, args.ckpt_path, args.figdir, args.layers)
    net.compare(args.filenames[0], args.filenames[1])


if __name__ == '__main__':
    main()

