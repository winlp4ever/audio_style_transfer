import tensorflow as tf
import matplotlib.pyplot as plt
from mdl import Cfg
import numpy as np
import use
import librosa
import argparse
import os

plt.switch_backend('agg')

def build_graph(length, lyr_stack=1, nb_channels=60):
    config = Cfg()
    with tf.device("/gpu:0"):
        x = tf.Variable(
            initial_value=(np.zeros([1, length])),
            trainable=True,
            name='regenerated_wav'
        )

        graph = config.build({'quantized_wav': x}, is_training=True)

        stl = []
        for i in range(nb_channels):
            embeds = tf.stack([config.extracts[j][0, :, i + 60] for j in range(lyr_stack * 10, lyr_stack * 10 + 10)], axis=1)
            embeds = tf.matmul(embeds, embeds, transpose_a=True) / length
            embeds = tf.nn.l2_normalize(embeds)
            stl.append(embeds)
        style_embeds = tf.stack(stl, axis=0)
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
    return sess.run(graph['embeds'], feed_dict={graph['quantized_input']: use.mu_law_numpy(aud)})

def read_file(filename, length, sr=16000):
    aud, _ = librosa.load(filename, sr=sr)
    auds = [aud[i * length: (i + 1) * length] for i in range(len(aud) // length)]
    return auds


def get_path(figdir, filename, stack):
    path = use.crt_t_fol(figdir)
    path = os.path.join(path, 'testshow::chan60-119f:{}stack{}'.format(filename, stack))
    if not os.path.exists(path):
        os.makedirs(path)
    return path

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
        figdir = get_path(self.figdir, fn, self.stack)

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            load_model(self.graph, sess, self.ckpt_path)

            embeds = [get_embeds(self.graph, sess, aud) for aud in audios]

            for i in range(len(embeds)):
                use.show_gram(embeds[i], i, figdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--srcdir', nargs='?', default='./data/src')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--stack', nargs='?', default=1, type=int)
    parser.add_argument('--channel', nargs='?', default=60, type=int)
    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')

    args = parser.parse_args()

    net = ShowNet(args.srcdir, args.ckpt_path, args.figdir, args.stack, args.channel)
    net.show(args.filename)


if __name__ == '__main__':
    main()

