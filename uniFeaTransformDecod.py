import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from model import cfg
import utils
import librosa
from nmf_matrixupdate_tensorflow import mynmf
import time
import argparse
import matplotlib.pyplot as plt
from nsynth.wavenet import fastgen
from numpy.linalg import norm, eigh


tf.logging.set_verbosity(tf.logging.WARN)

plt.switch_backend('agg')

class dvd_test(object):
    def __init__(self, savepath, checkpoint_path, logdir, figdir, batch_size=16384, sr=16000):
        self.logdir = logdir
        self.savepath = savepath
        self.checkpoint_path = checkpoint_path
        self.figdir = figdir
        self.batch_size = batch_size
        self.sr = sr
        self.graph, self.embeds = self.build(batch_size)

    @staticmethod
    def build(length):
        config = cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=(np.zeros([1, length])),
                trainable=True,
                name='regenerated_wav'
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

        embeds = tf.transpose(graph['encoding'][0])
        return graph, embeds

    def load_model(self, sess):
        variables = tf.global_variables()
        variables.remove(self.graph['quantized_input'])

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.checkpoint_path)

    def get_embeds(self, sess, aud):
        if len(aud.shape) == 1:
            aud = aud[: self.batch_size]
            aud = np.reshape(aud, [1, self.batch_size])
        embeds = sess.run(self.embeds,
                          feed_dict={self.graph['quantized_input']: utils.mu_law_numpy(aud)})

        return embeds

    def get_phi(self, sess, filename):
        print('load file ...')
        audio, _ = librosa.load(filename, sr=self.sr)
        I = []
        i = 0
        while i + self.batch_size < min(len(audio), 50 * self.batch_size):

            embeds = self.get_embeds(sess, audio[i: i + self.batch_size])
            I.append(embeds)
            print('I size {}'.format(len(I)), end='\r', flush=True)
            i += self.batch_size

        phi = np.mean(I, axis=0)
        print('style phi shape {}'.format(phi.shape))
        return phi

    def factorise(self, phi):
        m = np.mean(phi)
        phi_ = phi
        u = np.dot(phi_, phi_.T) + 1e-12
        w, v = eigh(u)
        w = np.sqrt(w)
        return m, w, v

    def matching(self, phic, phis, alpha=1.0):
        mc, Dc, Ec = self.factorise(phic)
        ms, Ds, Es = self.factorise(phis)

        phic_ = np.dot(np.dot(Ec, np.diag(1 / Dc)), Ec.T)
        phic_ = np.dot(phic_, phic)
        phisc = np.dot(np.dot(Es, np.diag(Ds)), Es.T)
        phisc = np.dot(phisc, phic_)
        return alpha * phisc + (1 - alpha) * phic

    def l_bfgs(self, sess, encodings, epochs, lambd):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        with tf.name_scope('loss'):
            loss = \
                (1 - lambd) * tf.nn.l2_loss(tf.concat(self.graph['encoding'], axis=2) - encodings)
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

        print('Saving file ... to fol {{{}}}'.format(self.savepath))
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
                utils.vis_actis(audio[0], enc, self.figdir, ep, [29])

            sp = os.path.join(self.savepath, 'ep-{}.wav'.format(ep))
            librosa.output.write_wav(sp, audio[0] / np.max(audio[0]), sr=self.sr)

    def run(self, src_file, trg_file, alpha=1.0, sample_length=32000):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            phis = self.get_phi(sess, trg_file)
            aud, _ = librosa.load(src_file, sr=self.sr)
            phic = self.get_embeds(sess, aud)

            phics = self.matching(phic, phis, alpha=alpha)

        print('phics shape {}'.format(phics.shape))
        fastgen.synthesize(
            np.expand_dims(phics.T, axis=0),
            save_paths=[os.path.join(self.savepath, 'synthesize.wav')],
            checkpoint_path=self.checkpoint_path,
            samples_per_save=sample_length)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('src_fn')
    parser.add_argument('trg_fn')
    parser.add_argument('--alpha', nargs='?', type=float, default=1.0)
    parser.add_argument('--batch_size', nargs='?', type=int, default=64000)
    parser.add_argument('--sr', nargs='?', type=int, default=16000)

    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--dir', nargs='?', default='./data/src')
    parser.add_argument('--outdir', nargs='?', default='./data/out')
    parser.add_argument('--logdir', nargs='?', default='./log')
    parser.add_argument('--cmt')


    args = parser.parse_args()

    savepath = utils.gt_s_path(utils.crt_t_fol(args.outdir), **vars(args))
    figdir = utils.gt_s_path(utils.crt_t_fol(args.figdir), **vars(args))
    logdir = utils.gt_s_path(utils.crt_t_fol(args.logdir), **vars(args))

    delta = lambda name: os.path.join(args.dir, name) + '.wav'

    src_fn = delta(args.src_fn)
    trg_fn = delta(args.trg_fn)

    test = dvd_test(savepath, args.ckpt_path, logdir, figdir, args.batch_size, args.sr)
    test.run(src_fn, trg_fn, args.alpha)

if __name__ == '__main__':
    main()








