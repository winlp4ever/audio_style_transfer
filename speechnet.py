import numpy as np
import librosa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import argparse
from spectrogram import plotstft
from rainbowgram import plotcqt
from model import Cfg
import matplotlib.pyplot as plt
from nmf_matrixupdate_tensorflow import mynmf
import utils

tf.logging.set_verbosity(tf.logging.WARN)

plt.switch_backend('agg')

#MALE = [17, 61, 81, 154, 562, 817, 866, 926, 1041, 1066, 1106, 1298, 1437,
#        1509, 1541, 1593]
#FEMALE = [419, 812, 1000, 1224, 1228, 1333, 1460, 1567, 1618]
MALE = [17]
FEMALE = [419]

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


def load_wav(path, l, s):
    if path:
        wav, _ = librosa.load(path, sr=s, mono=True)
        wav = wav[:l]
        return np.reshape(wav, [1, l])
    return None


class SpeechNet(object):
    def __init__(self, trg_path, src_path, spath, fig_dir,
                 tf_path='./data/dataset/aac-test.tfrecord',
                 checkpoint_path='./nsynth/model/wavenet-ckpt/model.ckpt-200000',
                 logdir='./log',
                 layers=[17],
                 sr=16000,
                 length=16384):
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


        src, trg = load_wav(src_path, length, sr), load_wav(trg_path, length, sr)

        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=(utils.mu_law_numpy(src) if src is not None
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
        if len(aud.shape) == 1:
            aud = np.reshape(aud, [1, self.length])
        embeds = sess.run(self.embeds,
                          feed_dict={self.graph['quantized_input']: utils.mu_law_numpy(aud)})
        embeds = np.concatenate(embeds, axis=2)
        return embeds

    def cpt_differ(self, sess, male2female, examples, n_components):
        it = self.data.make_one_shot_iterator()
        id, aud = it.get_next()

        I_m, I_f = [], []

        try:
            i, j, k = 0, 0, 0
            while True:
                i += 1
                id_, aud_ = sess.run([id, aud])
                aud_ = aud_[:self.length]

                if id_ in MALE and j < examples:
                    m_s = self.get_embeds(sess, aud_)
                    I_m.append(m_s)
                    j += 1

                elif id_ in FEMALE and k < examples:
                    m_t = self.get_embeds(sess, aud_)
                    I_f.append(m_t)
                    k += 1

                elif j == examples and k == examples:
                    break

                print('MALE - size {} -- FEMALE - size {} -- iter {}'.
                                format(j, k, i), end='\r', flush=True)

        except tf.errors.OutOfRangeError:
            pass

        # ============================== NMF ==============================
        f = lambda u: (np.concatenate(u, axis=1))[0].T
        if male2female:
            phi_s, phi_t = f(I_m), f(I_f)
        else:
            phi_s, phi_t = f(I_f), f(I_m)

        print('\nbegin nmf ...')

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

        i = 0

        def loss_tracking(loss_, summ_):
            nonlocal i_
            nonlocal i
            nonlocal ep
            nonlocal since
            if not i % 5:
                print('Ep: {0:}/{1:}--iter {2:} (last: {3:})--TOTAL time-lapse {4:.2f}s--loss: {5:.4f}'.
                      format(ep + 1, epochs, i, i_, time.time() - since, loss_), end='\r', flush=True)
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
            audio = utils.inv_mu_law_numpy(audio)

            if not (ep + 1) % 10:
                enc = self.get_embeds(sess, audio)
                utils.vis_actis(audio[0], enc, self.fig_dir, ep, self.layers)

            sp = os.path.join(self.spath, 'ep-{}.wav'.format(ep))
            librosa.output.write_wav(sp, audio[0] / np.max(audio[0]), sr=self.sr)

    def run(self, m2f, epochs, lambd, examples, n_components):
        assert 0 <= m2f <= 2

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            encodings = self.get_embeds(sess, self.wav)

            print('\nEnc shape: {}\n'.format(encodings.shape))
            if m2f < 2:
                ws, wt = self.cpt_differ(sess, m2f, examples, n_components)
                encodings = utils.transform(encodings, ws, wt, n_components, self.fig_dir)

            self.l_bfgs(sess, encodings, epochs, lambd)

    def test(self):
        fn1 = './data/src/male.wav'
        fn2 = './data/src/female.wav'
        wav1 = load_wav(fn1, self.length, self.sr)
        wav2 = load_wav(fn2, self.length, self.sr)
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)
            embeds1 = self.get_embeds(sess, wav1)
            embeds2 = self.get_embeds(sess, wav2)

            embeds1[:, :, :100] = embeds2[:, :, :100]
            self.l_bfgs(sess, embeds1, 100, 0.)


def main():
    class DefaultList(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 0:
                values = [30]
            setattr(namespace, self.dest, values)

    prs = argparse.ArgumentParser()

    prs.add_argument('filename', help='relative filename to transfer style.')
    prs.add_argument('male2female', help='source id', type=int)

    prs.add_argument('--n_components', help='number of components', nargs='?', default=20, type=int)
    prs.add_argument('--src_dir', help='dir where found files to be style-transferred',
                     nargs='?', default='./data/src')
    prs.add_argument('--src_name', help='relative path of source file to initiate with, if None the optim'
                                        'process will be initiated with zero vector')

    prs.add_argument('--out_dir', help='dir where stocks output files',
                     nargs='?', default='./data/out')

    prs.add_argument('--fig_dir', help='where stocks figures', nargs='?', default='./data/fig')

    prs.add_argument('-p', '--ckpt_path', help='checkpoint path', nargs='?',
                     default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    prs.add_argument('-t', '--tfpath', help='TFRecord Dataset s path', nargs='?',
                     default='./data/dataset/aac-test.tfrecord')
    prs.add_argument('--logdir', help='logging directory', nargs='?',
                     default='./log')
    prs.add_argument('-e', '--epochs', help='number of epochs', nargs='?', type=int, default=10)
    prs.add_argument('-l', '--lambd', help='lambda value', nargs='?', type=float, default=0.0)
    prs.add_argument('--length', help='duration of wav file -- unit: nb of samples', nargs='?',
                     type=int, default=16384)
    prs.add_argument('--sr', help='sampling rate', nargs='?', type=int, default=16000)
    prs.add_argument('--examples', help='number examples', nargs='?', type=int, default=1000)
    prs.add_argument('--layers', help='list of layer enums for embeddings', nargs='*',
                     type=int, action=DefaultList, default=[29])
    prs.add_argument('--cmt', help='comment')

    args = prs.parse_args()

    crt_path = lambda dir: utils.gt_s_path(utils.crt_t_fol(dir), **vars(args))
    savepath = crt_path(args.out_dir)
    logdir = crt_path(args.logdir)
    plotpath = crt_path(args.fig_dir)

    filepath = os.path.join(args.src_dir, args.filename + '.wav')

    if args.src_name:
        src_path = os.path.join(args.src_dir, args.src_name + '.wav')
    else:
        src_path = None

    net = SpeechNet(filepath, src_path, savepath, plotpath, args.tfpath, args.ckpt_path, logdir, args.layers, args.sr,
                    args.length)
    #net.test()
    net.run(args.male2female, args.epochs, args.lambd, args.examples, args.n_components)

    # save spec and cqt figs
    plotstft(os.path.join(savepath, 'ep-{}.wav'.format(args.epochs - 1)), plotpath=os.path.join(plotpath, 'spec.png'))
    plotcqt(os.path.join(savepath, 'ep-{}.wav'.format(args.epochs - 1)), savepath=os.path.join(plotpath, 'cqt.png'))


if __name__ == '__main__':
    main()
