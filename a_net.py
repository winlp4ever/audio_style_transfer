import numpy as np
import librosa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import time
import argparse
from spectrogram import plotstft
from rainbowgram import plotcqt
from mdl import Cfg
import matplotlib.pyplot as plt
from sklearn.decomposition.nmf import non_negative_factorization
from numpy.linalg import norm, lstsq
from optimal_transport import compute_permutation
from mynmf import mynmf
import use

tf.logging.set_verbosity(tf.logging.WARN)

plt.switch_backend('agg')

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


class Net(object):
    def __init__(self, trg_path, src_path, spath, fig_dir, tf_path, checkpoint_path, logdir, layers, sr, length):
        self.data = tf.data.TFRecordDataset([tf_path]).map(decode)
        self.checkpoint_path = checkpoint_path
        self.spath = spath
        self.fig_dir = fig_dir
        self.logdir = logdir
        self.length = length
        self.sr = sr
        self.layers = layers
        self.wav, self.graph, self.embeds = self.build(src_path, trg_path, layers, length, sr)

    @staticmethod
    def build(src_path, trg_path, layers, length, sr):
        def load_wav(path, l, s):
            if path:
                wav, _ = librosa.load(path, sr=s, mono=True)
                wav = wav[:l]
                return np.reshape(wav, [1, l])
            return None

        src, trg = load_wav(src_path, length, sr), load_wav(trg_path, length, sr)

        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=(use.mu_law_numpy(src) if src is not None
                               else np.zeros([1, length])),
                trainable=True,
                name='regenerated_wav'
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

        lyrs = [config.extracts[i] for i in layers]

        return trg, graph, lyrs

    def load_model(self, sess):
        variables = tf.global_variables()
        variables.remove(self.graph['quantized_input'])

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.checkpoint_path)

    def get_embeds(self, sess, aud):
        if len(aud.shape) < 2:
            aud = np.reshape(aud, [1, -1])

        embeds = sess.run(self.embeds,
                          feed_dict={self.graph['quantized_input']: use.mu_law_numpy(aud)})
        embeds = np.concatenate(embeds, axis=2)

        return embeds

    def cpt_differ(self, sess, type_s, type_t, qualities, examples, n_components):
        it = self.data.make_one_shot_iterator()
        id, src, qua, aud = it.get_next()

        I_s, I_t = [], []

        try:
            i, j, k = 0, 0, 0
            while True:
                i += 1
                id_, src_, qua_, aud_ = sess.run([id, src, qua, aud])
                aud_ = aud_[:self.length]

                if id_ == type_s and src_ == 0 and (qua_[qualities] == 1).all() and j < examples:
                    m_s = self.get_embeds(sess, aud_)
                    I_s.append(m_s)
                    j += 1

                elif id_ == type_t and src_ == 0 and (qua_[qualities] == 1).all() and k < examples:
                    m_t = self.get_embeds(sess, aud_)
                    I_t.append(m_t)
                    k += 1

                elif j == examples and k == examples:
                    break

                print('SRC: {} - size {} -- TRG: {} - size {} -- iter {}'.
                      format(use.ins[type_s], j, use.ins[type_t], k, i), end='\r', flush=True)

        except tf.errors.OutOfRangeError:
            pass

        # ============================== NMF ==============================
        f = lambda u: (np.concatenate(u, axis=1))[0].T
        phi_s, phi_t = f(I_s), f(I_t)

        print('\n begin nmf ...')

        ws, hs = mynmf(phi_s, n_components=n_components, epochs=1000)
        wt, ht = mynmf(phi_t, n_components=n_components, epochs=1000)

        return ws, wt

    def l_bfgs(self, sess, encodings, epochs, lambd):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        with tf.name_scope('loss'):
            loss = \
                (1 - lambd) * tf.nn.l2_loss(tf.concat(self.embeds, axis=2) - encodings)
            tf.summary.scalar('loss', loss)

        summ = tf.summary.merge_all()

        def loss_tracking(loss_, summ_):
            nonlocal i
            nonlocal i_
            nonlocal ep
            nonlocal since
            if not i % 5:
                print('Ep: {0:}/{1:}--iter {2:} (last: {3:})--TOTAL time-lapse {4:.2f}s--loss: {5:.4f}'.
                      format(ep, epochs - 1, i, i_, int(time.time() - since), loss_), end='\r', flush=True)
            writer.add_summary(summ_, global_step=i)
            i += 1

        with tf.name_scope('optim'):
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                var_list=[self.graph['quantized_input']],
                method='L-BFGS-B',
                options={'maxiter': 100})

        print('Saving file ... to fol {{{}}}'.format(self.spath))
        since = time.time()
        i_ = 0
        for ep in range(epochs):
            i = 0

            optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, summ])
            i_ = i
            audio = sess.run(self.graph['quantized_input'])
            audio = use.inv_mu_law_numpy(audio)

            if not (ep + 1) % 10:
                enc = self.get_embeds(sess, audio)
                use.vis_actis(audio[0], enc, self.fig_dir, ep, self.layers)

            sp = os.path.join(self.spath, 'ep-{}.wav'.format(ep))
            librosa.output.write_wav(sp, audio[0] / np.max(audio[0]), sr=self.sr)

    @staticmethod
    def regen_embeds(embeds):
        batch_size = 16

        rshpe = np.reshape(embeds, [1, -1, batch_size, 128])
        mean = np.mean(rshpe, axis=2)
        std = np.std(rshpe, axis=2)

        l = rshpe.shape[1]
        u = np.random.standard_normal([1, l, batch_size, 128])

        u = np.multiply(u, np.expand_dims(std, axis=2))
        u = np.add(u, np.expand_dims(mean, axis=2))

        return np.reshape(u, [1, -1, 128])

    def run(self, ins_families, qualities, epochs, lambd, examples, n_components):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            encodings = self.get_embeds(sess, self.wav)

            print('\nEnc shape: {}\n'.format(encodings.shape))
            if ins_families is not None:
                assert len(ins_families) == 2
                type_s, type_t = ins_families
                ws, wt = self.cpt_differ(sess, type_s, type_t, qualities, examples, n_components)
                encodings = use.transform(encodings, ws, wt, n_components)
            else:
                encodings = self.regen_embeds(encodings)

            self.l_bfgs(sess, encodings, epochs, lambd)


def main():
    class DefaultList(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 0:
                values = [30]
            setattr(namespace, self.dest, values)

    prs = argparse.ArgumentParser()

    prs.add_argument('filename', help='relative filename to transfer style.')
    prs.add_argument('--ins', help='source and target instrument families', nargs='*', type=int)
    prs.add_argument('--qualities', help='music note qualities', nargs='*', action=DefaultList,
                     default=[1], type=int)
    prs.add_argument('--n_components', help='number of components', nargs='?', type=int, default=20)
    prs.add_argument('--src_dir', help='dir where found files to be style-transferred',
                     nargs='?', default='./data/src')
    prs.add_argument('--src_name', help='relative path of source file to initiate with, if None the optim'
                                        'process will be initiated with zero vector')

    prs.add_argument('--out_dir', help='dir where stocks output files',
                     nargs='?', default='./data/out')

    prs.add_argument('--fig_dir', help='where stocks figures', nargs='?', default='./data/fig')

    prs.add_argument('--ckpt_path', help='checkpoint path', nargs='?',
                     default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    prs.add_argument('--tfpath', help='TFRecord Dataset s path', nargs='?',
                     default='./data/dataset/nsynth-train.tfrecord',
                     const='./data/dataset/nsynth-valid.tfrecord')
    prs.add_argument('--logdir', help='logging directory', nargs='?',
                     default='./log')
    prs.add_argument('--epochs', help='number of epochs', nargs='?', type=int, default=10)
    prs.add_argument('--lambd', help='lambda value', nargs='?', type=float, default=0.0)
    prs.add_argument('--length', help='duration of wav file -- unit: nb of samples', nargs='?',
                     type=int, default=16384)
    prs.add_argument('--sr', help='sampling rate', nargs='?', type=int, default=16000)
    prs.add_argument('--examples', help='number examples', nargs='?', type=int, default=1000)
    prs.add_argument('--layers', help='list of layer enums for embeddings', nargs='*',
                     type=int, action=DefaultList, default=[30])
    prs.add_argument('--cmt', help='comment')

    args = prs.parse_args()

    crt_path = lambda dir: use.gt_s_path(use.crt_t_fol(dir), **vars(args))
    savepath = crt_path(args.out_dir)
    logdir = crt_path(args.logdir)
    plotpath = crt_path(args.fig_dir)

    filepath = os.path.join(args.src_dir, args.filename + '.wav')

    if args.src_name:
        src_path = os.path.join(args.src_dir, args.src_name + '.wav')
    else:
        src_path = None

    net = Net(filepath, src_path, savepath, plotpath, args.tfpath, args.ckpt_path, logdir, args.layers, args.sr,
              args.length)
    net.run(args.ins, args.qualities, args.epochs, args.lambd, args.examples, args.n_components)

    # save spec and cqt figs
    plotstft(os.path.join(savepath, 'ep-{}.wav'.format(args.epochs - 1)), plotpath=os.path.join(plotpath, 'spec.png'))
    plotcqt(os.path.join(savepath, 'ep-{}.wav'.format(args.epochs - 1)), savepath=os.path.join(plotpath, 'cqt.png'))


if __name__ == '__main__':
    main()
