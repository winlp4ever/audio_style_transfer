from model import Cfg
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import librosa
import argparse
import time
from spectrogram import plotstft
import utils

plt.switch_backend('agg')

tf.logging.set_verbosity(tf.logging.WARN)

ACOUSTIC = 0
ELECTRONIC = 1
SYNTHETIC = 2


def gt_spath(suppath, ins_type, layers_ids, cmt=None):
    for id in layers_ids:
        ins_type += '_' + str(id)
    path = os.path.join(suppath, ins_type)
    if cmt:
        path += '_{}'.format(cmt)
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
    return ex['instrument_family'], ex['instrument_source'], ex['qualities'], ex['audio']


def mu_law(x, mu=255, int8=False):
    out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
    out = tf.floor(out * 128)
    if int8:
        out = tf.cast(out, tf.int8)
    return out


class ShowOff(object):
    def __init__(self, tfpath, ckptpath, figdir, layer_ids, length, sr):
        self.ins_fam = {'bass': 0,
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
        id, src, qua, aud = it.get_next()

        config = Cfg()
        with tf.device("/gpu:0"):
            x = mu_law(aud[:self.length])
            x = tf.reshape(x, shape=[1, self.length])

            graph = config.build({'quantized_wav': x}, is_training=True)

        layers = [config.extracts[i] for i in self.layer_ids]
        return id, src, qua, aud, graph, layers

    def load_model(self, sess):
        variables = tf.global_variables()

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.checkpoint_path)

    def run(self, ins_type, ins_src, qualities, examples, nb_channels, dspl, output_file, ens, cmt):
        assert 0 <= ins_src <= 2

        tpe = self.ins_fam[ins_type]

        figdir = gt_spath(self.figdir, ins_type, self.layer_ids, cmt)

        id, src, qua, aud, graph, layers = self.build()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            try:
                j = 0
                i = 0
                while i < examples:
                    id_, src_, qua_, aud_ = sess.run([id, src, qua, aud])

                    if tpe == id_ and src_ == ins_src and (qua_[qualities] == 1).all():
                        i += 1
                        #a = np.zeros((64000,))
                        #a[4096 : ] = aud_[:-4096]
                        #aud_=a
                        actis = sess.run(layers, feed_dict={
                            aud: aud_
                        })

                        actis = np.concatenate(actis, axis=0)
                        assert (actis >= 0).all()
                        if ens:
                            utils.vis_actis_ens(aud_, actis, figdir, i, self.layer_ids, nb_channels, dspl, output_file)
                        else:
                            utils.vis_actis(aud_, actis, figdir, i, self.layer_ids, nb_channels, dspl, output_file)

                    print('example {} -- iter {}'.format(i, j), end='\r', flush=True)
                    j += 1

            except tf.errors.OutOfRangeError:
                pass


def main():
    class DefaultList(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 0:
                values = [29]
            setattr(namespace, self.dest, values)

    prs = argparse.ArgumentParser()

    prs.add_argument('ins', help='instrument family')
    prs.add_argument('--source', '--instrument_source', help='instrument source', type=int,
                     nargs='?', default=0)
    prs.add_argument('--qualities', help='note qualities', type=int, nargs='*', action=DefaultList, default=[1])
    prs.add_argument('--tfpath', help='.tfrecord dataset path', nargs='?',
                     default='./data/dataset/nsynth-train.tfrecord')
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
    prs.add_argument('--examples', help='nb of examples', type=int,
                     nargs='?', default=100)
    prs.add_argument('--nb_channels', help='nb of first channels to be taken for each layer',
                     type=int, nargs='?', default=5)
    prs.add_argument('--dspl', '--downsampling-rate', help='downsampling rate', type=int,
                     nargs='?', default=64)
    prs.add_argument('--layers', help='layer ids', nargs='*', type=int, action=DefaultList,
                     default=[4, 9, 14, 19, 24, 29])
    prs.add_argument('--ens', help='view entirely or only partially original input signal',
                     nargs='?', default=False, const=True, type=bool)
    prs.add_argument('--cmt', help='comment')

    args = prs.parse_args()

    figdir = utils.crt_t_fol(args.figdir)
    showoff = ShowOff(args.tfpath, args.ckptpath, figdir, args.layers, args.length, args.sr)
    showoff.run(args.ins, args.source, args.qualities, args.examples, args.nb_channels, args.dspl, args.output_file,
                args.ens, args.cmt)


if __name__ == '__main__':
    main()
