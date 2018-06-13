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
from rainbowgram import plotcqtgram
from mdl import Cfg

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
    def __init__(self, trg_path, src_path, spath, tf_path, checkpoint_path, logdir, layers, sr, length):
        self.data = tf.data.TFRecordDataset([tf_path]).map(decode)
        self.checkpoint_path = checkpoint_path
        self.spath = spath
        self.logdir = logdir
        self.length = length
        self.sr = sr
        self.nb_layers = len(layers)
        self.wav, self.graph, self.layers = self.build(src_path, trg_path, layers, length, sr)

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
                initial_value=(mu_law_numpy(src) if src is not None
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

    def knear(self, sess, type_s, type_t, k):
        it = self.data.make_one_shot_iterator()
        el = it.get_next()

        N_s, N_t = MyHeap(k), MyHeap(k)

        encodings = sess.run([self.layers[i] for i in range(self.nb_layers)], feed_dict={
            self.graph['quantized_input']: mu_law_numpy(self.wav)
        })

        i = 0
        try:
            while True:
                i += 1
                ins = sess.run(el['instrument_family'])

                if ins == type_s:
                    audio = np.reshape(sess.run(el['audio'][:self.length]), [1, self.length])
                    enc = sess.run([self.layers[i] for i in range(self.nb_layers)], feed_dict={
                        self.graph['quantized_input']: mu_law_numpy(audio)})
                    dist = np.sum([norm(encodings[i] - enc[i]) for i in range(self.nb_layers)])

                    N_s.push((-dist, i, enc))
                    tf.logging.info(' sources - size {} - iterate {}'.format(len(N_s), i))

                elif ins == type_t:
                    audio = np.reshape(sess.run(el['audio'][:self.length]), [1, self.length])
                    enc = sess.run([self.layers[i] for i in range(self.nb_layers)], feed_dict={
                        self.graph['quantized_input']: mu_law_numpy(audio)})
                    dist = np.sum([norm(encodings[i] - enc[i]) for i in range(self.nb_layers)])

                    N_t.push((-dist, i, enc))
                    tf.logging.info(' targets - size {} - iterate {}'.format(len(N_t), i))
        except tf.errors.OutOfRangeError:
            pass

        sources = [[N_s[m][2][i] for m in range(len(N_s))] for i in range(self.nb_layers)]
        targets = [[N_t[m][2][i] for m in range(len(N_t))] for i in range(self.nb_layers)]

        for i in range(self.nb_layers):
            encodings[i] = np.mean(targets[i], axis=0)

        return encodings

    def l_bfgs(self, sess, encodings, epochs, lambd):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        with tf.name_scope('loss'):

            loss = \
                (1 - lambd) * tf.nn.l2_loss([(self.layers[i] - encodings[i]) for i in range(self.nb_layers)])
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
                var_list=[self.graph['quantized_input']],
                method='L-BFGS-B',
                options={'maxiter': 100})

        for ep in range(epochs):
            since = int(time.time())

            optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, summ])
            tf.logging.info(' Saving file ... Epoch: {} -- time-lapse: {}s'.format(ep, int(time.time() - since)))

            audio = sess.run(self.graph['quantized_input'])
            audio = inv_mu_law_numpy(audio)
            librosa.output.write_wav(self.spath + '_' + str(ep) + '.wav', audio.T, sr=self.sr)

    def run(self, type_s, type_t, k, epochs, lambd):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            if type_s != type_t:
                encodings = self.knear(sess, type_s, type_t, k)
            else:
                encodings = sess.run([self.layers[i] for i in range(self.nb_layers)], feed_dict={
                    self.graph['quantized_input']: mu_law_numpy(self.wav)
                })

            self.l_bfgs(sess, encodings, epochs, lambd)


def main():
    class DefaultList(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 0:
                values = [30]
            setattr(namespace, self.dest, values)

    prs = argparse.ArgumentParser()

    prs.add_argument('fname', help='relative filename to transfer style.')
    prs.add_argument('s', help='source type', type=int)
    prs.add_argument('t', help='target type', type=int)

    prs.add_argument('--src_name', help='relative path of source file to initiate with, if None the optim'
                                        'process will be initiated with zero vector')

    prs.add_argument('-p', '--ckpt_path', help='checkpoint path', nargs='?',
                     default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    prs.add_argument('-t', '--tfpath', help='TFRecord Dataset s path', nargs='?',
                     default='./data/dataset/nsynth-valid.tfrecord')
    prs.add_argument('--logdir', help='logging directory', nargs='?',
                     default='./log')
    prs.add_argument('--k', help='nb of nearest neighbors', nargs='?', type=int, default=10)
    prs.add_argument('-e', '--epochs', help='number of epochs', nargs='?', type=int, default=10)
    prs.add_argument('-l', '--lambd', help='lambda value', nargs='?', type=float, default=0.0)
    prs.add_argument('--length', help='duration of wav file -- unit: nb of samples', nargs='?',
                     type=int, default=16384)
    prs.add_argument('--sr', help='sampling rate', nargs='?', type=int, default=16000)
    prs.add_argument('--layers', help='list of layer enums for embeddings', nargs='*',
                     type=int, action=DefaultList, default=[30])
    prs.add_argument('--cmt', help='comment', nargs='?', default='')

    args = prs.parse_args()

    save_name = get_s_name(args.s, args.t, args.fname, args.k, args.lambd, args.cmt)
    save_path = os.path.join(crt_fol('./data/out/'), save_name)
    filepath = os.path.join('./data/src/', args.fname + '.wav')

    if args.src_name:
        src_path = os.path.join('./data/src/', args.src_name + '.wav')
    else:
        src_path = None

    logdir = crt_fol(args.logdir, True)
    plot_path = os.path.join(crt_fol('./data/out/fig'), save_name)

    net = Net(filepath, src_path, save_path, args.tfpath, args.ckpt_path, logdir, args.layers, args.sr, args.length)
    net.run(args.s, args.t, args.k, args.epochs, args.lambd)

    plotstft('{}_{}.wav'.format(save_path, args.epochs - 1), plotpath='{}_spec.png'.format(plot_path))
    plotcqtgram('{}_{}.wav'.format(save_path, args.epochs - 1), savepath='{}_cqt.png'.format(plot_path))


if __name__ == '__main__':
    main()
