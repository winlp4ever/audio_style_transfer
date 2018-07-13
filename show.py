import tensorflow as tf
import matplotlib.pyplot as plt
from mdl import Cfg
import numpy as np
import use
import librosa
import argparse
import os

plt.switch_backend('agg')

def build_graph(length, lyr_stack=1, channel=0):
    config = Cfg()
    with tf.device("/gpu:0"):
        x = tf.Variable(
            initial_value=(np.zeros([1, length])),
            trainable=True,
            name='regenerated_wav'
        )

        graph = config.build({'quantized_wav': x}, is_training=True)

        embeds = tf.stack([config.extracts[i][0, :, channel] for i in range(lyr_stack * 10, lyr_stack * 10 + 10)], axis=1)
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

def compare2embeds(embeds1, embeds2, stack, figdir=None, ord=None):
    fig, axs = plt.subplots(1, 2, figsize=(30, 10))
    ptp = np.dot(embeds1.T, embeds1)
    ete = np.dot(embeds2.T, embeds2)
    axs[0].imshow(ptp, interpolation='nearest', cmap=plt.cm.plasma)
    axs[1].imshow(ete, interpolation='nearest', cmap=plt.cm.plasma)
    axs[0].set_title('{}'.format(stack))
    assert figdir, 'figdir must not be None'
    assert ord is not None
    plt.savefig(os.path.join(figdir, 'p{}'.format(ord)))


def get_path(figdir, fn1, fn2, stack, channel=0):
    path = use.crt_t_fol(figdir)
    path = os.path.join(path, 'compare::chan{}{}-{}stack{}'.format(channel, fn1, fn2, stack))
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class ShowNet(object):
    def __init__(self, srcdir, ckpt_path, figdir, stack, channel=0, length=16384, sr=16000):
        self.graph = build_graph(length, stack, channel)
        self.srcdir = srcdir
        assert ckpt_path, 'must provide a ckpt path for this model!'
        self.ckpt_path = ckpt_path
        self.figdir = figdir
        self.sr = sr
        self.length = length
        self.stack = stack
        self.channel = channel

    def compare(self, fn1, fn2):
        fp1 = os.path.join(self.srcdir, fn1 + '.wav')
        fp2 = os.path.join(self.srcdir, fn2 + '.wav')

        auds1 = read_file(fp1, self.length)
        auds2 = read_file(fp2, self.length)
        figdir = get_path(self.figdir, fn1, fn2, self.stack, self.channel)

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            load_model(self.graph, sess, self.ckpt_path)

            embeds1 = [get_embeds(self.graph, sess, aud) for aud in auds1]
            embeds2 = [get_embeds(self.graph, sess, aud) for aud in auds2]

            for i in range(min(len(embeds1), len(embeds2))):
                compare2embeds(embeds1[i], embeds2[i], self.stack, figdir, i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    parser.add_argument('--srcdir', nargs='?', default='./data/src')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--stack', nargs='*', default=1, type=int)
    parser.add_argument('--channel', nargs='?', default=1, type=int)
    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')

    args = parser.parse_args()

    net = ShowNet(args.srcdir, args.ckpt_path, args.figdir, args.stack, args.channel)
    net.compare(args.filenames[0], args.filenames[1])


if __name__ == '__main__':
    main()

