import tensorflow as tf
from mdl import Cfg
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import argparse
import time
plt.switch_backend('agg')

tf.logging.set_verbosity(tf.logging.INFO)

def crt_t_fol(suppath, hour=False):
    dte = time.localtime()
    if hour:
        fol_n = os.path.join(suppath, '{}{}{}{}'.format(dte[1], dte[2], dte[3], dte[4]))
    else:
        fol_n = os.path.join(suppath, '{}{}'.format(dte[1], dte[2]))

    if not os.path.exists(fol_n):
        os.makedirs(fol_n)
    return fol_n

def gt_spath(suppath, ins_type, layers_ids):
    for id in layers_ids:
        ins_type += '_' + str(id)
    path = os.path.join(suppath, ins_type)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def decode(serialized_example):
    ex = tf.parse_single_example(
        serialized_example,
        features={
            "note_str": tf.FixedLenFeature([], dtype=tf.string),
            "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
            "velocity": tf.FixedLenFeature([1], dtype=tf.int64),
            "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
            "qualities": tf.FixedLenFeature([10], dtype=tf.int64),
            "instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
            "instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
        }
    )
    return ex['instrument_family'], ex['audio']


def mu_law(x, mu=255, int8=False):
    out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
    out = tf.floor(out * 128)
    if int8:
        out = tf.cast(out, tf.int8)
    return out


class ShowOff(object):
    def __init__(self, tfpath, ckptpath, figdir, layer_ids, length, sr):
        self.map = {'bass': 0,
                    'brass': 1,
                    'flute': 2,
                    'guitar': 3,
                    'keyboard': 4,
                    'mallet': 5,
                    'organ': 6,
                    'reed': 7,
                    'string': 8,
                    'synth_lead': 9,
                    'vocal': 10}
        self.data = tf.data.TFRecordDataset([tfpath]).map(decode)
        self.checkpoint_path = ckptpath
        self.figdir = figdir
        self.length = length
        self.sr = sr
        self.layer_ids = layer_ids

    def build(self):
        it = self.data.make_one_shot_iterator()
        id, aud = it.get_next()

        config = Cfg()
        with tf.device("/gpu:0"):
            x = mu_law(aud[:self.length])
            x = tf.reshape(x, shape=[1, self.length])

            graph = config.build({'quantized_wav': x}, is_training=True)

        layers = [config.extracts[i] for i in self.layer_ids]
        return id, aud, graph, layers

    def load_model(self, sess):
        variables = tf.global_variables()

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.checkpoint_path)

    @staticmethod
    def vis_actis(aud, enc, fig_dir, ep, layer_ids, nb_channels=5, dspl=256):
        nb_layers = enc.shape[0]
        fig, axs = plt.subplots(nb_layers + 1, 1, figsize=(10, 5 * nb_layers))
        axs[0].plot(aud)
        axs[0].set_title('Audio Signal')
        for i in range(nb_layers):
            axs[i + 1].plot(enc[i, :: dspl, :nb_channels])
            axs[i + 1].set_title('Embeds layer {}'.format(layer_ids[i]))

        plt.savefig(os.path.join(fig_dir, 'f-{}.png'.format(ep)), dpi=50)

    def run(self, ins_type, nb_exs, nb_channels, dspl):
        type = self.map[ins_type]

        figdir = gt_spath(self.figdir, ins_type, self.layer_ids)

        id, aud, graph, layers = self.build()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                i = 0
                while i < nb_exs:
                    id_, aud_, actis = sess.run([id, aud, layers])
                    actis = np.concatenate(actis, axis=0)
                    if type == id_:
                        self.vis_actis(aud_, actis, figdir, i, self.layer_ids, nb_channels, dspl)
                        tf.logging.info('example {}'.format(i))
                        i += 1

            except tf.errors.OutOfRangeError:
                pass
        tf.logging.info('Done')

def main():
    class DefaultList(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 0:
                values = [30]
            setattr(namespace, self.dest, values)

    prs = argparse.ArgumentParser()

    prs.add_argument('ins', help='instrument family')
    prs.add_argument('--tfpath', help='.tfrecord dataset path', nargs='?',
                     default='./data/dataset/nsynth-valid.tfrecord')
    prs.add_argument('--ckptpath', help='checkpoint path', nargs='?',
                     default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    prs.add_argument('--figdir', help='where to store figures', nargs='?',
                     default='./data/fig')
    prs.add_argument('--length', help='file duration to trim from beginning', type=int,
                     nargs='?', default=16384)
    prs.add_argument('--sr', '--samplingrate', help='sampling rate', type=int,
                     nargs='?', default=16000)
    prs.add_argument('--nb_exs', help='nb of examples', type=int,
                     nargs='?', default=100)
    prs.add_argument('--nb_channels', help='nb of first channels to be taken for each layer',
                     type=int, nargs='?', default=5)
    prs.add_argument('--dspl', '--downsampling-rate', help='downsampling rate', type=int,
                     nargs='?', default=256)
    prs.add_argument('--layers', help='layer ids', nargs='*', type=int, action=DefaultList,
                     default=[4, 9, 14, 19, 24, 29])

    args= prs.parse_args()

    figdir = crt_t_fol(args.figdir)

    showoff = ShowOff(args.tfpath, args.ckptpath, figdir, args.layers, args.length, args.sr)
    showoff.run(args.ins, args.nb_exs, args.nb_channels, args.dspl)


if __name__ == '__main__':
    main()
