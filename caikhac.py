import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from mod import Cfg
import use
import librosa
from mynmf import mynmf
import time
import argparse
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.WARN)

plt.switch_backend('agg')

class GatysNet(object):
    def __init__(self,
                 savepath='./data/out',
                 checkpoint_path='./nsynth/model/wavenet-ckpt/model.ckpt-200000',
                 logdir='./log',
                 figdir='./data/fig',
                 stack=1,
                 batch_size=16384,
                 sr=16000,
                 cont_lyr_ids=[29]):
        self.logdir = logdir
        self.savepath = savepath
        self.checkpoint_path = checkpoint_path
        self.figdir = figdir
        self.batch_size = batch_size
        self.sr = sr
        self.cont_lyr_ids = cont_lyr_ids
        self.graph, self.embeds_c, self.embeds_s = self.build(batch_size, cont_lyr_ids, stack)

    @staticmethod
    def build(length, cont_lyr_ids, stack=1):
        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=tf.abs(tf.random_normal(shape=[1, length])),
                trainable=True,
                name='regenerated_wav',
                dtype=tf.float32
            )

            graph = config.build({'wav': x}, is_training=True)
            graph.update({'input': x})

        cont_embeds = tf.concat([config.extracts[i] for i in cont_lyr_ids], axis=2)[0]
        stl = []
        for i in range(60):
            embeds = tf.stack([config.extracts[j][0, :, i] for j in range(stack * 10, stack * 10 + 10)], axis=1)
            embeds = tf.matmul(embeds, embeds, transpose_a=True)
            embeds = tf.nn.l2_normalize(embeds)
            stl.append(embeds)

        style_embeds = tf.stack(stl, axis=0)

        return graph, cont_embeds, style_embeds

    def load_model(self, sess):
        variables = tf.global_variables()
        variables.remove(self.graph['input'])

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
                 feed_dict={self.graph['input']: aud})

    def get_style_phi(self, sess, filename, max_examples=3, show_mat=True):
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
        if show_mat:
            use.show_gram(phi, figdir=self.figdir)
        return phi

    def l_bfgs(self, sess, phi_c, phi_s, epochs, lambd, gamma):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        abso = lambda u: tf.maximum(u, 1e-12) + tf.maximum(0.0, -u)

        with tf.name_scope('loss'):
            content_loss = tf.nn.l2_loss(self.embeds_c - phi_c)
            style_loss = tf.nn.l2_loss(self.embeds_s - phi_s)
            style_loss *= 1e2
            regularizer = tf.contrib.signal.stft(self.graph['input'][0], frame_length=1024, frame_step=512, name='stft')
            regularizer = tf.reduce_mean(abso(tf.real(regularizer)) + abso(tf.imag(regularizer)))
            regularizer *= 1e3

            loss = content_loss + lambd * style_loss + gamma * regularizer

            tf.summary.scalar('content_loss', content_loss)
            tf.summary.scalar('style_loss', style_loss)
            tf.summary.scalar('regularizer', regularizer)
            tf.summary.scalar('main_loss', loss)

        summ = tf.summary.merge_all()

        def loss_tracking(loss_, cont_loss_, style_loss_, regularizer_, summ_):
            nonlocal i_
            nonlocal i
            nonlocal ep
            nonlocal since
            if not i % 5:
                print('Ep {0:}/{1:}-it {2:}({3:})-tlapse {4:.2f}s-loss{5:.2f}-{6:.2f}-{7:.2f}-{8:.2f}'.
                      format(ep + 1, epochs, i, i_, time.time() - since, loss_, cont_loss_, style_loss_, regularizer_), end='\r', flush=True)
            writer.add_summary(summ_, global_step=i_ + i)
            i += 1

        with tf.name_scope('optim'):
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                var_list=[self.graph['input']],
                method='L-BFGS-B',
                options={'maxiter': 100})

        print('Saving file ... to fol {{{}}}'.format(self.savepath))
        since = time.time()
        i_ = 0
        for ep in range(epochs):
            i = 0

            optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, content_loss, style_loss, regularizer, summ])
            i_ = i
            audio = sess.run(self.graph['input'])
            #audio = use.inv_mu_law_numpy(audio)

            sp = os.path.join(self.savepath, 'ep-{}.wav'.format(ep))
            librosa.output.write_wav(sp, audio[0] / np.max(audio[0]), sr=self.sr)

            gram = sess.run(self.embeds_s)
            use.show_gram(gram, ep, self.figdir)

    def run(self, cont_file, style_file, epochs=20000, lambd=0.1, gamma=0.1):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            phi_s = self.get_style_phi(sess, style_file)
            aud, _ = librosa.load(cont_file, sr=self.sr)
            phi_c = self.get_embeds(sess, aud)

            self.l_bfgs(sess, phi_c, phi_s, epochs=epochs, lambd=lambd, gamma=gamma)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cont_fn')
    parser.add_argument('style_fn')
    parser.add_argument('--epochs', nargs='?', type=int, default=100)
    parser.add_argument('--batch_size', nargs='?', type=int, default=16384)
    parser.add_argument('--sr', nargs='?', type=int, default=16000)
    parser.add_argument('--cont_lyrs', nargs='*', type=int, default=[29])
    parser.add_argument('--stack', nargs='?', type=int, default=1)
    parser.add_argument('--lambd', nargs='?', type=float, default=0.1)
    parser.add_argument('--gamma', nargs='?', type=float, default=0.0)

    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--dir', nargs='?', default='./data/src')
    parser.add_argument('--outdir', nargs='?', default='./data/out')
    parser.add_argument('--logdir', nargs='?', default='./log')
    parser.add_argument('--cmt')

    args = parser.parse_args()

    savepath, figdir, logdir = map(lambda dir: use.gt_s_path(use.crt_t_fol(dir), 'caikhac', **vars(args)),
                                   [args.outdir, args.figdir, args.logdir])

    content, style = map(lambda name: os.path.join(args.dir, name) + '.wav', [args.cont_fn, args.style_fn])

    test = GatysNet(savepath, args.ckpt_path, logdir, figdir, args.stack, args.batch_size, args.sr, args.cont_lyrs)
    test.run(content, style, epochs=args.epochs, lambd=args.lambd, gamma=args.gamma)


if __name__ == '__main__':
    main()