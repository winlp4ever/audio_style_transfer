import tensorflow as tf
import numpy as np
import librosa
import os
import time
import argparse
from spectrogram import plotstft
from rainbowgram import plotcqt
from mdl import Cfg
import matplotlib.pyplot as plt

plt.switch_backend('agg')

tf.logging.set_verbosity(tf.logging.INFO)


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


def crt_t_fol(suppath, hour=False):
    dte = time.localtime()
    if hour:
        fol_n = os.path.join(suppath, '{}{}{}{}'.format(dte[1], dte[2], dte[3], dte[4]))
    else:
        fol_n = os.path.join(suppath, '{}{}'.format(dte[1], dte[2]))

    if not os.path.exists(fol_n):
        os.makedirs(fol_n)
    return fol_n


def gt_s_path(suppath, s, t, fname, l, b, y, r, cmt):
    if s != t:
        path = '{}2{}_{}_l{}_b{}_y'.format(s, t, fname, l, b)
    else:
        path = '{}_l{}_b{}_y'.format(fname, l, b)
    for i in y:
        path += str(i)
    if r:
        path += '_fr{}'.format(r)
    if cmt:
        path += '_{}'.format(cmt)
    path = os.path.join(suppath, path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


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


class SpeechNet(object):
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

    def get_embeds(self, sess, aud):
        embeds = sess.run(self.embeds,
                          feed_dict={self.graph['quantized_input']: mu_law_numpy(aud)})
        embeds = np.concatenate(embeds, axis=0)
        return embeds

    def dvd_embeds(self, sess, aud, batch_size=512):
        if len(aud.shape) == 1:
            aud = np.reshape(aud, [1, self.length])

        enc = self.get_embeds(sess, aud)

        assert self.length % batch_size == 0
        nb_batches = self.length // batch_size

        pieces = np.split(enc, nb_batches, axis=1)
        mean = np.mean(pieces, axis=0)

        return mean

    def cpt_differ(self, sess, id_s, id_t, batch_size, nb_exs):
        it = self.data.make_one_shot_iterator()
        id, aud = it.get_next()

        I_s, I_t = 0, 0

        try:
            i, j, k = 0, 0, 0
            while True:
                i += 1
                id_, aud_ = sess.run([id, aud])
                aud_ = aud_[:self.length]

                if id_ == id_s and j < nb_exs:
                    m_s = self.dvd_embeds(sess, aud_, batch_size)
                    I_s = (j * I_s + m_s) / (j + 1)
                    j += 1

                elif id_ == id_t and k < nb_exs:
                    m_t = self.dvd_embeds(sess, aud_, batch_size)
                    I_t = (k * I_t + m_t) / (k + 1)
                    k += 1

                elif j == nb_exs and k == nb_exs:
                    break

                tf.logging.info(' SRC: {} - size {} -- TRG: {} - size {} -- iter {}'.
                                format(id_s, j, id_t, k, i))

        except tf.errors.OutOfRangeError:
            pass

        w = I_t - I_s

        return w

    @staticmethod
    def transform(enc, w, length, batch_size):
        a = [w for _ in range(length // batch_size)]
        a = np.concatenate(a, axis=1)
        return enc + a

    @staticmethod
    def vis_actis(aud, enc, fig_dir, ep, layers, nb_channels=5, dspl=256):
        nb_layers = enc.shape[0]
        fig, axs = plt.subplots(nb_layers + 1, 3, figsize=(30, 10 * nb_layers))
        axs[0, 1].plot(aud)
        axs[0, 1].set_title('Audio Signal')
        axs[0, 0].axis('off')
        axs[0, 2].axis('off')
        for i in range(nb_layers):
            axs[i + 1, 0].plot(enc[i, :dspl, :nb_channels])
            axs[i + 1, 0].set_title('Embeds layer {} part 0'.format(layers[i]))
            axs[i + 1, 1].plot(enc[i, dspl:2 * dspl, :nb_channels])
            axs[i + 1, 1].set_title('Embeds layer {} part 1'.format(layers[i]))
            axs[i + 1, 2].plot(enc[i, 2 * dspl:3 * dspl, :nb_channels])
            axs[i + 1, 2].set_title('Embeds layer {} part 2'.format(layers[i]))
        plt.savefig(os.path.join(fig_dir, 'f-{}.png'.format(ep)), dpi=50)

    def l_bfgs(self, sess, encodings, epochs, lambd):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        with tf.name_scope('loss'):
            loss = \
                (1 - lambd) * tf.nn.l2_loss(tf.concat(self.embeds, axis=0) - encodings)
            tf.summary.scalar('loss', loss)

        summ = tf.summary.merge_all()

        i = 0

        def loss_tracking(loss_, summ_):
            nonlocal i
            if not i % 5:
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
            tf.logging.info(' Saving file ... to fol {{{}}} \n \t Epoch: {}/{} -- Time-lapse: {}s'.
                            format(self.spath, ep, epochs - 1, int(time.time() - since)))

            audio = sess.run(self.graph['quantized_input'])
            audio = inv_mu_law_numpy(audio)

            if not (ep + 1) % 10:
                tf.logging.info(' visualize actis ...')
                enc = self.get_embeds(sess, audio)
                self.vis_actis(audio[0], enc, self.fig_dir, ep, self.layers)

            sp = os.path.join(self.spath, 'ep-{}.wav'.format(ep))
            librosa.output.write_wav(sp, audio[0], sr=self.sr)

    def run(self, id_s, id_t, epochs, lambd, batch_size, nb_exs):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            encodings = self.get_embeds(sess, self.wav)

            tf.logging.info('\nEnc shape: {}\n'.format(encodings.shape))
            if id_s != id_t:
                w = self.cpt_differ(sess, id_s, id_t, batch_size, nb_exs)
                encodings = self.transform(encodings, w, self.length, batch_size)

            self.l_bfgs(sess, encodings, epochs, lambd)


def main():
    class DefaultList(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 0:
                values = [30]
            setattr(namespace, self.dest, values)

    prs = argparse.ArgumentParser()

    prs.add_argument('filename', help='relative filename to transfer style.')
    prs.add_argument('s', help='source id', type=int)
    prs.add_argument('t', help='target id', type=int)

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
    prs.add_argument('--batch_size', help='batch size', nargs='?', type=int, default=512, const=16384)
    prs.add_argument('--nb_exs', help='number examples', nargs='?', type=int, default=1000)
    prs.add_argument('--layers', help='list of layer enums for embeddings', nargs='*',
                     type=int, action=DefaultList, default=[30])
    prs.add_argument('--cmt', help='comment')

    args = prs.parse_args()

    s, t, fn, l, b, y, r, cmt, e = args.s, args.t, args.filename, args.lambd, \
                                   args.batch_size, args.layers, args.src_name, args.cmt, args.epochs

    savepath = crt_t_fol(args.out_dir)
    savepath = gt_s_path(savepath, s, t, fn, l, b, y, r, cmt)

    filepath = os.path.join(args.src_dir, fn + '.wav')

    if args.src_name:
        src_path = os.path.join(args.src_dir, args.src_name + '.wav')
    else:
        src_path = None

    logdir = crt_t_fol(args.logdir)
    logdir = gt_s_path(logdir, s, t, fn, l, b, y, r, cmt)
    plotpath = crt_t_fol(args.fig_dir)
    plotpath = gt_s_path(plotpath, s, t, fn, l, b, y, r, cmt)

    net = SpeechNet(filepath, src_path, savepath, plotpath, args.tfpath, args.ckpt_path, logdir, args.layers, args.sr,
                    args.length)
    net.run(s, t, e, l, b, args.nb_exs)

    # save spec and cqt figs
    plotstft(os.path.join(savepath, 'ep-{}.wav'.format(e - 1)), plotpath=os.path.join(plotpath, 'spec.png'))
    plotcqt(os.path.join(savepath, 'ep-{}.wav'.format(e - 1)), savepath=os.path.join(plotpath, 'cqt.png'))


if __name__ == '__main__':
    main()
