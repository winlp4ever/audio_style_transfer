import tensorflow as tf
import matplotlib.pyplot as plt
from model import Cfg
import numpy as np
import utils
import librosa
import argparse
import os
from numpy.linalg import norm

plt.switch_backend('agg')

ARR = [0, 5, 6, 7, 10, 21, 22, 29, 30, 32, 34, 39, 41,
       42, 46, 47, 49, 53, 58, 59, 62, 63, 65, 66, 68, 69,
       71, 72, 73, 74, 76, 78, 80, 81, 84, 85, 86, 87, 90,
       93, 96, 97, 100, 101, 102, 103, 105, 107, 109, 110, 112, 113,
       114, 119, 127]

def build_graph(length, lyr_stack=1, nb_channels=128):
    config = Cfg()
    with tf.device("/gpu:0"):
        x = tf.Variable(
            initial_value=(np.zeros([1, length])),
            trainable=True,
            name='regenerated_wav'
        )

        graph = config.build({'quantized_wav': x}, is_training=True)

        if lyr_stack is not None:
            stl = tf.concat([config.extracts[i] for i in range(lyr_stack * 10, lyr_stack * 10 + 10)], axis=0)
        else:
            stl = tf.concat([config.extracts[i] for i in range(30)], axis=0)
        stl = tf.transpose(stl, perm=[2, 0, 1])

        style_embeds = tf.matmul(stl, tf.transpose(stl, perm=[0, 2, 1]))
        style_embeds = tf.nn.l2_normalize(style_embeds, axis=(1, 2))
        if nb_channels < 128:
            style_embeds = style_embeds[:nb_channels]
        graph.update({'embeds': style_embeds})

    return graph

def load_model(graph, sess, checkpoint_path):
    vars = tf.global_variables()
    vars.remove(graph['quantized_input'])

    saver = tf.train.Saver(vars)
    saver.restore(sess, checkpoint_path)

def get_embeds(graph, sess, aud):
    if len(aud.shape) == 1:
        aud = np.reshape(aud, [1, -1])
    return sess.run(graph['embeds'], feed_dict={graph['quantized_input']: utils.mu_law_numpy(aud)})

def read_file(filename, length, sr=16000):
    aud, _ = librosa.load(filename, sr=sr)
    auds = [aud[i * length: (i + 1) * length] for i in range(len(aud) // length)]
    return auds


def get_path(figdir, filename, stack, length):
    path = utils.crt_t_fol(figdir)
    path = os.path.join(path, 'showAcrosslayer::chan0-127f:{}stack{}length{}'.format(filename, stack, length))
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def show_inten(mats, ep, figdir):
    nb_ch = mats.shape[0]
    print(np.shape(mats))
    a = np.array([norm(mats[i]) for i in range(nb_ch)])
    print(np.where(a >= 2))
    print(np.where(a >= 2)[0].shape)
    plt.plot(a)
    plt.savefig(os.path.join(figdir, 'int{}'.format(ep)), dpi=100, clear=True)
    plt.close()

class ShowNet(object):
    def __init__(self, srcdir, ckpt_path, figdir, stack, channels=60, length=16384, sr=16000):
        self.graph = build_graph(length, stack, channels)
        self.srcdir = srcdir
        assert ckpt_path, 'must provide a ckpt path for this model!'
        self.ckpt_path = ckpt_path
        self.figdir = figdir
        self.sr = sr
        self.length = length
        self.stack = stack
        self.channels = channels

    def show(self, fn):
        filepath = os.path.join(self.srcdir, fn + '.wav')

        audios = read_file(filepath, self.length)
        figdir = get_path(self.figdir, fn, self.stack, self.length)

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            load_model(self.graph, sess, self.ckpt_path)

            embeds = [get_embeds(self.graph, sess, aud) for aud in audios]

            for i in range(len(embeds)):
                utils.show_gram(embeds[i], i, figdir)
                #show_inten(embeds[i], i, figdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--srcdir', nargs='?', default='./data/src')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--stack', nargs='?', default=None, type=int)
    parser.add_argument('--channels', nargs='?', default=128, type=int)
    parser.add_argument('--length', nargs='?', default=16384, type=int)
    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')

    args = parser.parse_args()

    net = ShowNet(args.srcdir, args.ckpt_path, args.figdir, args.stack, args.channels, args.length)
    net.show(args.filename)


if __name__ == '__main__':
    main()

