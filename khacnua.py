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
from numpy.linalg import norm, eigh

tf.logging.set_verbosity(tf.logging.WARN)

ARR = [0, 5, 6, 7, 10, 21, 22, 29, 30, 32, 34, 39, 41,
       42, 46, 47, 49, 53, 58, 59, 62, 63, 65, 66, 68, 69,
       71, 72, 73, 74, 76, 78, 80, 81, 84, 85, 86, 87, 90,
       93, 96, 97, 100, 101, 102, 103, 105, 107, 109, 110, 112, 113,
       114, 119, 127]

plt.switch_backend('agg')

class GatysNet(object):
    def __init__(self,
                 savepath='./data/out',
                 checkpoint_path='./nsynth/model/wavenet-ckpt/model.ckpt-200000',
                 logdir='./log',
                 figdir='./data/fig',
                 stack=1,
                 batch_size=16384,
                 sr=16000):
        self.logdir = logdir
        self.savepath = savepath
        self.checkpoint_path = checkpoint_path
        self.figdir = figdir
        self.batch_size = batch_size
        self.sr = sr
        self.graph, self.embeds = self.build(batch_size, stack)

    @staticmethod
    def build(length, stack):
        config = cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=np.zeros([1, length]) + 1e-12,
                trainable=True,
                name='regenerated_wav',
                dtype=tf.float32
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

            stl = []
            for i in range(128):
                embd = tf.stack([config.extracts[j][0, :, i] for j in range(stack * 10, stack * 10 + 10)], axis=0)
                stl.append(embd)

            embeds = tf.stack(stl, axis=0)

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

        return sess.run(self.embeds,
                        feed_dict={self.graph['quantized_input']: utils.mu_law_numpy(aud)})

    def get_style_phi(self, sess, filename, max_examples=3):
        print('load file ...')
        audio, _ = utils.load_audio(filename, sr=self.sr, audio_channel=0)
        I = []
        i = 0
        while i + self.batch_size <= min(len(audio), max_examples * self.batch_size):

            embeds = self.get_embeds(sess, audio[i: i + self.batch_size])
            I.append(embeds)
            print('I size {}'.format(len(I)), end='\r', flush=True)
            i += self.batch_size

        phi = np.mean(I, axis=0)
        print('style phi shape {}'.format(phi.shape))
        return phi

    def factorise(self, phi):
        m = np.mean(phi)
        phi_ = phi - m
        u = np.dot(phi_, phi_.T) + 1e-12
        w, v = eigh(u)
        w = np.sqrt(w)
        return m, w, v

    def matching_each(self, pc, ps, alpha=1.0):
        mc, Dc, Ec = self.factorise(pc)
        ms, Ds, Es = self.factorise(ps)

        phic_ = np.dot(np.dot(Ec, np.diag(1 / Dc)), Ec.T)
        phic_ = np.dot(phic_, pc - mc)
        phisc = np.dot(np.dot(Es, np.diag(Ds)), Es.T)
        phisc = np.dot(phisc, phic_) + ms
        return alpha * phisc + (1 - alpha) * pc

    def matching_all(self, phic, phis, alpha=1.0):
        nb_ch = phic.shape[0]
        phics = phis.copy()
        for i in range(nb_ch):
            phics[i] = self.matching_each(phic[i], phis[i], alpha=alpha)
        return phics

    def l_bfgs(self, sess, phi_sc, epochs, gamma):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        #writer.add_graph(sess.graph)

        with tf.name_scope('loss'):
            scloss = tf.nn.l2_loss(self.embeds - phi_sc)
            #stloss *= 1e2

            a = utils.inv_mu_law(self.graph['quantized_input'][0])
            regularizer = tf.contrib.signal.stft(a, frame_length=1024, frame_step=512, name='stft')
            regularizer = tf.reduce_mean(utils.abs(tf.real(regularizer)) + utils.abs(tf.imag(regularizer)))
            regularizer *= 1e3
            loss = scloss + gamma * regularizer

            tf.summary.scalar('sc_loss', scloss)
            tf.summary.scalar('regularizer', regularizer)
            tf.summary.scalar('main_loss', loss)

        summ = tf.summary.merge_all()

        def loss_tracking(loss_, scloss, regularizer_, summ_):
            nonlocal i_
            nonlocal i
            nonlocal ep
            nonlocal since
            if not i % 5:
                print('Ep {0:}/{1:}-it {2:}({3:})-tlapse {4:.2f}s-loss{5:.2f}-{6:.2f}-{7:.2f}'.
                      format(ep + 1, epochs, i, i_, time.time() - since, loss_, scloss, regularizer_), end='\r', flush=True)
            writer.add_summary(summ_, global_step=i_ + i)
            i += 1

        with tf.name_scope('optim'):
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                var_list=[self.graph['quantized_input']],
                method='L-BFGS-B',
                options={'maxiter': 100})

        writer.add_graph(sess.graph)

        print('Saving file ... to fol {{{}}}'.format(self.savepath))
        since = time.time()
        i_ = 0
        for ep in range(epochs):
            i = 0

            optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, scloss, regularizer, summ])
            i_ = i
            audio = sess.run(self.graph['quantized_input'])
            audio = utils.inv_mu_law_numpy(audio)

            #audio_test = sess.run(a)

            sp = os.path.join(self.savepath, 'ep-{}.wav'.format(ep))
            librosa.output.write_wav(sp, audio[0] / np.max(audio[0]), sr=self.sr)
            #sp = os.path.join(self.savepath, 'ep-test-{}.wav'.format(ep))
            #librosa.output.write_wav(sp, audio_test / np.mean(audio_test), sr=self.sr)

    def run(self, cont_file, style_file, epochs, alpha=1.0, gamma=0.1, piece=0):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            phi_s = self.get_style_phi(sess, style_file)
            aud, _ = utils.load_audio(cont_file, sr=self.sr, audio_channel=0)
            aud = aud[self.batch_size * piece: ]
            phi_c = self.get_embeds(sess, aud)

            phi_sc = self.matching_all(phi_c, phi_s, alpha=alpha)

            self.l_bfgs(sess, phi_sc, epochs=epochs, gamma=gamma)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cont_fn')
    parser.add_argument('style_fn')
    parser.add_argument('--epochs', nargs='?', type=int, default=100)
    parser.add_argument('--batch_size', nargs='?', type=int, default=16384)
    parser.add_argument('--sr', nargs='?', type=int, default=16000)
    parser.add_argument('--stack', nargs='?', type=int, default=1)
    parser.add_argument('--alpha', nargs='?', type=float, default=1.0)
    parser.add_argument('--gamma', nargs='?', type=float, default=0.00)
    parser.add_argument('--piece', nargs='?', type=int, default=0)

    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--dir', nargs='?', default='./data/src')
    parser.add_argument('--outdir', nargs='?', default='./data/out')
    parser.add_argument('--logdir', nargs='?', default='./log')
    parser.add_argument('--cmt')

    args = parser.parse_args()

    savepath, figdir, logdir = map(lambda dir: utils.gt_s_path(utils.crt_t_fol(dir), 'khacnua', **vars(args)),
                                   [args.outdir, args.figdir, args.logdir])

    content, style = map(lambda name: os.path.join(args.dir, name) + '.wav', [args.cont_fn, args.style_fn])

    test = GatysNet(savepath, args.ckpt_path, logdir, figdir, args.stack, args.batch_size, args.sr)
    test.run(content, style, epochs=args.epochs, alpha=args.alpha, gamma=args.gamma, piece=args.piece)


if __name__ == '__main__':
    main()