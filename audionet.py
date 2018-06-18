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
from rainbowgram import plotcqt
from mdl import Cfg
from nsynth.wavenet import masked
from geter import decode

tf.logging.set_verbosity(tf.logging.INFO)

INS = ['bass', 'brass', 'flute', 'guitar',
       'keyboard', 'mallet', 'organ', 'reed',
       'string', 'synth_lead', 'vocal']


def crt_fol(suppath, hour=False):
    dte = time.localtime()
    if hour:
        fol_n = os.path.join(suppath, '{}{}{}{}'.format(dte[1], dte[2], dte[3], dte[4]))
    else:
        fol_n = os.path.join(suppath, '{}{}'.format(dte[1], dte[2]))
    if not os.path.exists(fol_n):
        os.makedirs(fol_n)
    return fol_n


def get_s_name(s, t, fname, k, l, cmt):
    if s != t:
        return '{}2{}_{}_k{}_l{}{}_e'.format(INS[s], INS[t], fname, k, l, cmt)
    return '{}_k{}_l{}{}_e'.format(fname, k, l, cmt)


def mu_law_numpy(x, mu=255):
    out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    out = np.floor(out * 128)
    return out


def inv_mu_law_numpy(x, mu=255.0):
    x = np.array(x).astype(np.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu) ** np.abs(out) - 1)
    out = np.where(np.equal(x, 0), x, out)
    return out


class Net(object):
    def __init__(self, fpath, spath, tf_path, checkpoint_path, logdir, sr, length):
        self.data = tf.data.TFRecordDataset([tf_path]).map(decode)
        self.checkpoint_path = checkpoint_path
        self.spath = spath
        self.logdir = logdir
        self.length = length
        self.sr = sr
        self.wavs, self.graph = self.build(fpath, length, sr)

    @staticmethod
    def build(fpath, length, sr, batch_size=2 ** 12):
        assert length % batch_size == 0
        stride = batch_size // 2
        wavs = []
        i = 0
        while i < length - batch_size + 1:
            wav, _ = librosa.load(fpath, sr=sr, mono=True)
            wav = wav[i:i + batch_size]
            wavs.append(wav)
            i += stride
        wavs = np.stack(wavs, axis=0)

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

        nb_frags, batch_size = self.wavs.shape

        N_s = [MyHeap(k) for _ in range(nb_frags)]
        N_t = [MyHeap(k) for _ in range(nb_frags)]

        encodings = np.concatenate([sess.run(self.graph['before_enc'], feed_dict={
            self.graph['quantized_input']: mu_law_numpy(self.wavs[i:i + 1, ])
        }) for i in range(nb_frags)], axis=0)

        i = 0
        try:
            while True:
                i += 1
                ins = sess.run(el['instrument_family'])

                if ins == type_s:
                    audio = np.reshape(sess.run(el['audio'][:batch_size]), [1, batch_size])
                    enc = sess.run(self.graph['before_enc'], feed_dict={
                        self.graph['quantized_input']: mu_law_numpy(audio)})

                    for j in range(nb_frags):
                        N_s[j].push((-norm(enc - encodings[j]), i, enc))

                    tf.logging.info(' sources - size {} - iterate {}'.format(len(N_s[0]), i))

                elif ins == type_t:
                    audio = np.reshape(sess.run(el['audio'][:batch_size]), [1, batch_size])
                    enc = sess.run(self.graph['before_enc'], feed_dict={
                        self.graph['quantized_input']: mu_law_numpy(audio)})

                    for j in range(nb_frags):
                        N_t[j].push((-norm(enc - encodings[j]), i, enc))

                    tf.logging.info(' targets - size {} - iterate {}'.format(len(N_t[0]), i))
        except tf.errors.OutOfRangeError:
            pass

        sources = np.concatenate([np.mean(heap.as_list(), axis=0) for heap in N_s], axis=0)
        targets = np.concatenate([np.mean(heap.as_list(), axis=0) for heap in N_t], axis=0)

        beta = 0.4
        w = targets - sources

        alpha = beta / (w ** 2)
        return encodings + w

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
        coefs_1 = np.array([[[j / (stride - 1) if j < stride else 1 for _ in range(nb_channels)] for j in
                             range(batch_size)]])
        coefs_2 = np.array(
            [[[j / (stride - 1) if j < stride else (batch_size - 1 - j) / (stride - 1) for _ in
               range(nb_channels)] for j in range(batch_size)]])

        tf.logging.info('\n coefs shape: {}\n'.format(coefs_0.shape))

        encoding[:, 0: batch_size] = encodings[0] * coefs_0
        for i in range(1, nb_frags - 1):
            encoding[:, i * stride: i * stride + batch_size] += encodings[i] * coefs_2
        encoding[:, l - batch_size:] += encodings[nb_frags - 1] * coefs_1

        return encoding

    @staticmethod
    def load_model(sess, graph, checkpoint_path):
        variables = tf.global_variables()
        variables.remove(graph['quantized_input'])

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, checkpoint_path)

    def l_bfgs(self, encoding, epochs, lambd):
        tf.reset_default_graph()

        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=(np.zeros([1, self.length])),
                trainable=True,
                name='regenerated_wav',
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

        writer = tf.summary.FileWriter(logdir=self.logdir)

        with tf.name_scope('loss'):
            loss = \
                (1 - lambd) * tf.nn.l2_loss(graph['before_enc'] - encoding)
            tf.summary.scalar('loss', loss)

        summ = tf.summary.merge_all()

        i = 0

        def loss_tracking(loss_, summ_):
            nonlocal i
            tf.logging.info(' Step: {} -- Loss: {}'.format(i, loss_))
            writer.add_summary(summ_, global_step=i)
            i += 1

        with tf.name_scope('optim'):
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                var_list=[graph['quantized_input']],
                method='L-BFGS-B',
                options={'maxiter': 100})

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_model(sess, graph, self.checkpoint_path)

            writer.add_graph(sess.graph)

            for ep in range(epochs):
                since = int(time.time())

                optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, summ])
                tf.logging.info(' Saving file ... Epoch: {} -- time-lapse: {}s'.format(ep, int(time.time() - since)))

                audio = sess.run(graph['quantized_input'])
                audio = inv_mu_law_numpy(audio)
                librosa.output.write_wav(self.spath + '_' + str(ep) + '.wav', audio.T, sr=self.sr)

    def run(self, type_s, type_t, k, epochs, lambd):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess, self.graph, self.checkpoint_path)

            if type_s != type_t:
                encodings = self.knear(sess, type_s, type_t, k)
            else:
                nb_frags = self.wavs.shape[0]
                encodings = np.concatenate([sess.run(self.graph['before_enc'], feed_dict={
                    self.graph['quantized_input']: mu_law_numpy(self.wavs[i:i + 1, ])
                }) for i in range(nb_frags)], axis=0)

            encoding = self.get_encoding(encodings)

        self.l_bfgs(encoding, epochs, lambd)


def main():
    prs = argparse.ArgumentParser()

    prs.add_argument('fname', help='relative filename to transfer style.')
    prs.add_argument('s', help='source type', type=int)
    prs.add_argument('t', help='target type', type=int)

    prs.add_argument('-p', '--ckpt_path', help='checkpoint path', nargs='?',
                     default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    prs.add_argument('--tfpath', help='TFRecord Dataset s path', nargs='?',
                     default='./data/dataset/nsynth-valid.tfrecord')
    prs.add_argument('--logdir', help='logging directory', nargs='?',
                     default='./log')
    prs.add_argument('--k', help='nb of nearest neighbors', nargs='?', type=int, default=10)
    prs.add_argument('-e', '--epochs', help='number of epochs', nargs='?', type=int, default=10)
    prs.add_argument('-l', '--lambd', help='lambda value', nargs='?', type=float, default=0.0)
    prs.add_argument('--length', help='duration of wav file -- unit: nb of samples', nargs='?',
                     type=int, default=16384)
    prs.add_argument('--sr', help='sampling rate', nargs='?', type=int, default=16000)
    prs.add_argument('--cmt', help='comment', nargs='?', default='')

    args = prs.parse_args()

    save_name = get_s_name(args.s, args.t, args.fname, args.k, args.lambd, args.cmt)
    save_path = os.path.join(crt_fol('./data/out/'), save_name)
    filepath = os.path.join('./data/src/', args.fname + '.wav')

    logdir = crt_fol(args.logdir, True)
    plot_path = os.path.join(crt_fol('./data/out/fig'), save_name)

    net = Net(filepath, save_path, args.tfpath, args.ckpt_path, logdir, args.sr, args.length)
    net.run(args.s, args.t, args.k, args.epochs, args.lambd)

    plotstft('{}_{}.wav'.format(save_path, args.epochs - 1), plotpath='{}_spec.png'.format(plot_path))
    plotcqt('{}_{}.wav'.format(save_path, args.epochs - 1), savepath='{}_cqt.png'.format(plot_path))


if __name__ == '__main__':
    main()
