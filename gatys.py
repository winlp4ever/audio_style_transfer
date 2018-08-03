import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from mdl import Cfg
import use
import librosa
from mynmf import mynmf
import time
import argparse
import spectrogram
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.WARN)

plt.switch_backend('agg')

class GatysNet(object):
    def __init__(self,
                 savepath='./data/out',
                 checkpoint_path='./nsynth/model/wavenet-ckpt/model.ckpt-200000',
                 logdir='./log',
                 figdir='./data/fig',
                 batch_size=16384,
                 sr=16000,
                 cont_lyr_ids=[29],
                 style_lyr_ids=[9]):
        self.logdir = logdir
        self.savepath = savepath
        self.checkpoint_path = checkpoint_path
        self.figdir = figdir
        self.batch_size = batch_size
        self.sr = sr
        self.late = (batch_size - (batch_size // 4096) * 4000) // 2
        self.cont_lyr_ids = cont_lyr_ids
        self.style_lyr_ids = style_lyr_ids
        self.graph, self.embeds_c, self.embeds_s = self.build(batch_size, cont_lyr_ids, style_lyr_ids)

    @staticmethod
    def build(length, cont_lyr_ids, style_lyr_ids):
        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=(np.zeros([1, length])),
                trainable=True,
                name='regenerated_wav'
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

        cont_embeds = tf.concat([config.extracts[i] for i in cont_lyr_ids], axis=2)[0]
        style_embeds = tf.concat([config.extracts[i] for i in style_lyr_ids], axis=0)
        style_embeds = tf.matmul(tf.transpose(style_embeds, perm=[0, 2, 1]), style_embeds)
        style_embeds = tf.nn.l2_normalize(style_embeds, axis=[1, 2])

        return graph, cont_embeds, style_embeds

    def load_model(self, sess):
        variables = tf.global_variables()
        variables.remove(self.graph['quantized_input'])

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.checkpoint_path)

    def get_embeds(self, sess, aud, is_content=True):
        if len(aud.shape) == 1:
            aud = aud[: self.batch_size]
            aud = np.reshape(aud, [1, self.batch_size])
        if is_content:
            embeds = self.embeds_c
        else:
            embeds = self.embeds_s
        return sess.run(embeds,
                 feed_dict={self.graph['quantized_input']: use.mu_law_numpy(aud)})

    def get_style_phi(self, sess, filename, max_examples=10, show_mats=True):
        print('load file ...')
        audio, _ = librosa.load(filename, sr=self.sr)
        I = []
        i = 0
        while i + self.batch_size <= min(len(audio), max_examples * self.batch_size):

            embeds = self.get_embeds(sess, audio[i: i + self.batch_size], is_content=False)
            I.append(embeds)
            print('I size {}'.format(len(I)), end='\r', flush=True)
            i += self.batch_size

        phi = np.mean(I, axis=0)
        if show_mats:
            use.show_gatys_gram(phi, figdir=self.figdir)
        return phi

    def l_bfgs(self, sess, phi_c, phi_s, epochs, lambd, gamma):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        with tf.name_scope('loss'):
            content_loss = tf.losses.mean_squared_error(predictions=self.embeds_c, labels=phi_c)
            style_loss = tf.losses.mean_squared_error(predictions=self.embeds_s, labels=phi_s)
            style_loss *= 1e4

            a = use.inv_mu_law(self.graph['quantized_input'][0])
            regularizer = tf.contrib.signal.stft(a, frame_length=1024, frame_step=512, name='stft')
            regularizer = tf.reduce_mean(use.abs(tf.real(regularizer)) + use.abs(tf.imag(regularizer)))
            #regularizer *= 1e3
            loss = content_loss + lambd * style_loss + gamma * regularizer

            tf.summary.scalar('content_loss', content_loss)
            tf.summary.scalar('style_loss', style_loss)
            tf.summary.scalar('main_loss', loss)

        summ = tf.summary.merge_all()

        def loss_tracking(loss_, cont_loss_, style_loss_, regularizer_, summ_):
            nonlocal s
            nonlocal i_
            nonlocal i
            nonlocal ep
            nonlocal since
            if not i % 5:
                print('Ep {0:}/{1:}-it{2:}({3:})-tlapse {4:.2f}s-loss{5:.2f}-{6:.4f}-{7:.4f}-{8:.4f}'.
                      format(ep + 1, epochs, i, i_, time.time() - since, loss_, cont_loss_, style_loss_, regularizer_), end='\r', flush=True)
            writer.add_summary(summ_, global_step=s + i)
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
        s = 0
        for ep in range(epochs):
            i = 0

            optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, content_loss, style_loss, regularizer, summ])
            i_ = i
            s += i
            audio = sess.run(self.graph['quantized_input'])
            audio = use.inv_mu_law_numpy(audio)
            audio = audio[0, self.late:-self.late]

            sp = os.path.join(self.savepath, 'ep-{}.wav'.format(ep))
            librosa.output.write_wav(sp, audio / np.max(audio), sr=self.sr)

            if not ep + 1% 1 or i_ < 50:
                gram = sess.run(self.embeds_s)
                use.show_gatys_gram(gram, ep + 1, self.figdir)
                spectrogram.plotstft(sp, plotpath=os.path.join(self.figdir, 'ep_{}_spectro.png'.format(ep+1)))

            if i_ < 50:
                break
        print('\n')

    def run(self, cont_file, style_file, epochs, lambd=0.1, gamma=0.0, start=1.0):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            phi_s = self.get_style_phi(sess, style_file)
            aud, _ = librosa.load(cont_file, sr=self.sr)
            st = int(start * self.sr - self.late)
            aud = aud[st: st + self.batch_size]
            savep = os.path.join(self.savepath, 'ori.wav')
            librosa.output.write_wav(savep, aud[self.late:-self.late], sr=self.sr)
            spectrogram.plotstft(savep, plotpath=os.path.join(self.figdir, 'ori-spec.png'))
            phi_c = self.get_embeds(sess, aud)
            phi = self.get_embeds(sess, aud, is_content=False)
            use.show_gatys_gram(phi, ep=0, figdir=self.figdir)

            self.l_bfgs(sess, phi_c, phi_s, epochs=epochs, lambd=lambd, gamma=gamma)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cont_fn')
    parser.add_argument('style_fn')

    parser.add_argument('--epochs', nargs='?', type=int, default=100)
    parser.add_argument('--batch_size', nargs='?', type=int, default=16384)
    parser.add_argument('--sr', nargs='?', type=int, default=16000)
    parser.add_argument('--style_lyrs', nargs='*', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--cont_lyrs', nargs='*', type=int, default=[27])
    parser.add_argument('--lambd', nargs='?', type=float, default=0.1)
    parser.add_argument('--gamma', nargs='?', type=float, default=0.0)
    parser.add_argument('--start', nargs='?', type=float, default=1.0)

    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--dir', nargs='?', default='./data/src')
    parser.add_argument('--outdir', nargs='?', default='./data/out')
    parser.add_argument('--logdir', nargs='?', default='./log')
    parser.add_argument('--cmt')

    args = parser.parse_args()

    savepath, figdir, logdir = map(lambda dir: use.gt_s_path(use.crt_t_fol(dir), 'gatys', **vars(args)),
                                   [args.outdir, args.figdir, args.logdir])

    content, style = map(lambda name: os.path.join(args.dir, name) + '.wav', [args.cont_fn, args.style_fn])

    test = GatysNet(savepath, args.ckpt_path, logdir, figdir, args.batch_size, args.sr, args.cont_lyrs, args.style_lyrs)
    test.run(content, style, epochs=args.epochs, lambd=args.lambd, gamma=args.gamma, start=args.start)


if __name__ == '__main__':
    main()
