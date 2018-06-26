import tensorflow as tf
from mdl import Cfg
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import argparse
import time
from spectrogram import plotstft

MALE = [17, 61, 81, 154, 562, 817, 866, 926, 1041, 1066, 1106, 1298, 1437,
        1509, 1541, 1593]
FEMALE = [419, 812, 1000, 1224, 1228, 1333, 1460, 1567, 1618]


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


def gt_spath(suppath, male, layers_ids):
    s = 'male' if male else 'female'
    for id in layers_ids:
        s += '_' + str(id)
    path = os.path.join(suppath, s)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'id': tf.FixedLenFeature([], dtype=tf.int64),
            'audio': tf.FixedLenFeature([], dtype=tf.string)
        }
    )

    id = tf.cast(features['id'], tf.int32)

    audio = tf.decode_raw(features['audio'], tf.float32)
    return id, audio


def mu_law(x, mu=255, int8=False):
    out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
    out = tf.floor(out * 128)
    if int8:
        out = tf.cast(out, tf.int8)
    return out


class ShowOff(object):
    def __init__(self, tfpath, ckptpath, figdir, layer_ids, length, sr):

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
    def vis_actis(aud, enc, fig_dir, ep, layer_ids, nb_channels=5, dspl=256, output_file=False):
        nb_layers = enc.shape[0]
        fig, axs = plt.subplots(nb_layers + 1, 3, figsize=(30, 5 * (nb_layers + 1)))
        axs[0, 1].plot(aud)
        axs[0, 1].set_title('Audio Signal')
        axs[0, 0].axis('off')
        axs[0, 2].axis('off')
        for i in range(nb_layers):
            axs[i + 1, 0].plot(enc[i, :dspl, :nb_channels])
            axs[i + 1, 0].set_title('Embeds layer {} part 0'.format(layer_ids[i]))
            axs[i + 1, 1].plot(enc[i, dspl:2 * dspl, :nb_channels])
            axs[i + 1, 1].set_title('Embeds layer {} part 1'.format(layer_ids[i]))
            axs[i + 1, 2].plot(enc[i, 2 * dspl:3 * dspl, :nb_channels])
            axs[i + 1, 2].set_title('Embeds layer {} part 2'.format(layer_ids[i]))

        sp = os.path.join(fig_dir, 'f-{}'.format(ep))
        plt.savefig(sp+'.png', dpi=50)
        if output_file:
            librosa.output.write_wav(sp + '.wav', aud, sr=16000)

    @staticmethod
    def vis_actis_ens(aud, enc, fig_dir, ep, layer_ids, nb_channels=5, dspl=256, output_file=False):
        nb_layers = enc.shape[0]
        fig, axs = plt.subplots(nb_layers + 1, 3, figsize=(30, 5 * (nb_layers + 1)))
        axs[0, 1].plot(aud)
        axs[0, 1].set_title('Audio Signal')
        axs[0, 0].axis('off')
        axs[0, 2].axis('off')

        for i in range(nb_layers):
            a = np.reshape(enc[i, :, :nb_channels], [-1, dspl, nb_channels])
            std = np.std(a, axis=1)
            mean = np.mean(a, axis=1)
            min = np.std(a, axis=1)
            max = np.std(a, axis=1)
            axs[i + 1, 0].plot(min)
            axs[i + 1, 0].plot(max)
            axs[i + 1, 0].set_title('embeds layer {} -- MIN/MAX'.format(layer_ids[i]))
            axs[i + 1, 1].plot(std + mean)
            axs[i + 1, 1].plot(-std + mean)
            axs[i + 1, 1].set_title('embeds layer {} -- STD/MEAN'.format(layer_ids[i]))
            axs[i + 1, 2].plot(mean)
            axs[i + 1, 2].set_title('embeds layer {} -- AVG'.format(layer_ids[i]))

        sp = os.path.join(fig_dir, 'fe-{}'.format(ep))
        plt.savefig(sp + '.png', dpi=50)
        if output_file:
            librosa.output.write_wav(sp + '.wav', aud, sr=16000)

    def run(self, male, nb_exs, nb_channels, dspl, output_file, ens):
        s = MALE if male else FEMALE
        figdir = gt_spath(self.figdir, male, self.layer_ids)

        id, aud, graph, layers = self.build()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            try:
                j = 0
                i = 0
                while i < nb_exs:
                    id_, aud_ = sess.run([id, aud])

                    if id_ in s:
                        i += 1
                        actis = sess.run(layers, feed_dict={
                            aud: aud_
                        })
                        actis = np.concatenate(actis, axis=0)
                        if ens:
                            self.vis_actis_ens(aud_, actis, figdir, i, self.layer_ids, nb_channels, dspl, output_file)
                        else:
                            self.vis_actis(aud_, actis, figdir, i, self.layer_ids, nb_channels, dspl, output_file)

                    tf.logging.info('example {} -- iter {}'.format(i, j))
                    j += 1

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

    prs.add_argument('--male', help='male or female', nargs='?', type=bool, default=False, const=True)
    prs.add_argument('--tfpath', help='.tfrecord dataset path', nargs='?',
                     default='./data/dataset/aac-test.tfrecord')
    prs.add_argument('--ckptpath', help='checkpoint path', nargs='?',
                     default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    prs.add_argument('--figdir', help='where to store figures', nargs='?',
                     default='./data/fig')
    prs.add_argument('--output_file', help='choose if output src file or not', nargs='?', type=bool,
                     default=False, const=True)
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
    prs.add_argument('--ens', help='view entirely or only partially original input signal',
                     nargs='?', default=False, const=True, type=bool)

    args = prs.parse_args()

    figdir = crt_t_fol(args.figdir)
    showoff = ShowOff(args.tfpath, args.ckptpath, figdir, args.layers, args.length, args.sr)
    showoff.run(args.male, args.nb_exs, args.nb_channels, args.dspl, args.output_file, args.ens)


if __name__ == '__main__':
    main()
