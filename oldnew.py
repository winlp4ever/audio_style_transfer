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
import matplotlib.pyplot as plt
import spectrogram
from sklearn.preprocessing import normalize

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
                 sr=16000,
                 cont_lyr_ids=[29],
                 nb_channels=60,
                 cnt_channels=128):
        self.logdir = logdir
        self.savepath = savepath
        self.checkpoint_path = checkpoint_path
        self.figdir = figdir
        self.batch_size = batch_size
        self.sr = sr
        self.late = (batch_size - (batch_size // 4096) * 4000) // 2
        self.cont_lyr_ids = cont_lyr_ids
        self.graph, self.embeds_c, self.embeds_s0, self.embeds_s1 = self.build(batch_size, cont_lyr_ids, stack, nb_channels,
                                                               cnt_channels)

    @staticmethod
    def build(length, cont_lyr_ids, stack, nb_channels, cnt_channels=128):
        tf.reset_default_graph()
        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=np.zeros([1, length]) + 1e-12,
                trainable=True,
                name='regenerated_wav',
                dtype=tf.float32
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

            cont_embeds = tf.concat([config.extracts[i][:, :, :cnt_channels] for i in cont_lyr_ids], axis=2)[0]

            if stack is not None:
                stl = tf.concat([config.extracts[i] for i in range(stack * 10, stack * 10 + 10)], axis=0)
            else:
                stl = tf.concat([config.extracts[i] for i in range(30)], axis=0)

            style_embeds0 = tf.nn.l2_normalize(tf.matmul(tf.transpose(stl, perm=[0, 2, 1]), stl), axis=(1, 2))
            stl = tf.transpose(stl, perm=[2, 0, 1])

            style_embeds1 = tf.matmul(stl, tf.transpose(stl, perm=[0, 2, 1]))
            style_embeds1 = tf.nn.l2_normalize(style_embeds1, axis=(1, 2))
            if nb_channels < 128:
                style_embeds1 = style_embeds1[:nb_channels]
        return graph, cont_embeds, style_embeds0, style_embeds1

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
            embeds = [self.embeds_s0, self.embeds_s1]
        return sess.run(embeds,
                        feed_dict={self.graph['quantized_input']: use.mu_law_numpy(aud)})

    def get_style_phi(self, sess, filename, max_examples=30, show_mat=True):
        print('load file ...')
        audio, _ = use.load_audio(filename, sr=self.sr, audio_channel=0)
        I_0 = []
        I_1 = []
        i = 0
        while i + self.batch_size <= min(len(audio), max_examples * self.batch_size):
            embeds0, embeds1 = self.get_embeds(sess, audio[i: i + self.batch_size], is_content=False)
            I_0.append(embeds0)
            I_1.append(embeds1)
            print('I size {}'.format(len(I_0)), end='\r', flush=True)
            i += self.batch_size

        phi0 = np.mean(I_0, axis=0)
        phi1 = np.mean(I_1, axis=0)
        if show_mat:
            use.show_gram(phi1, figdir=self.figdir)
        return phi0, phi1

    def define_loss(self, name, stl_emb, cnt_emb, lambd, gamma, gpu):
        with tf.device(gpu):
            with tf.name_scope(name):
                content_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.embeds_c, labels=cnt_emb))
                style_loss0 = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.embeds_s0, labels=stl_emb[0]))
                style_loss0 *= 1e3

                style_loss1 = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.embeds_s1, labels=stl_emb[1]))
                style_loss1 *= 1e3

                a = use.inv_mu_law(self.graph['quantized_input'][0])
                regularizer = tf.contrib.signal.stft(a, frame_length=1024, frame_step=512, name='stft')
                regularizer = tf.reduce_mean(use.abs(tf.real(regularizer)) + use.abs(tf.imag(regularizer)))
                #regularizer *= 1e3
                loss = content_loss + lambd * (style_loss0 + style_loss1) + gamma * regularizer

                tf.summary.scalar('content_loss', content_loss)
                tf.summary.scalar('style_loss', style_loss0)
                tf.summary.scalar('regularizer', regularizer)
                tf.summary.scalar('main_loss', loss)

            with tf.name_scope('optim'):
                optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    loss,
                    var_list=[self.graph['quantized_input']],
                    method='L-BFGS-B',
                    options={'maxiter': 100})
        return content_loss, style_loss0, regularizer, loss, optimizer

    def l_bfgs(self, sess, phi_c, phi_s, epochs, lambd, gamma):
        writer = tf.summary.FileWriter(logdir=self.logdir)

        cnt_l, stl_l, regu, loss, optim = self.define_loss('loss', phi_s, phi_c, lambd, gamma, '/gpu:0')

        summ = tf.summary.merge_all()

        def loss_tracking(loss_, cont_loss_, style_loss_, regularizer_, summ_):
            nonlocal i_
            nonlocal i
            nonlocal ep
            nonlocal since
            if not i % 5:
                print('Ep {0:}/{1:}-it {2:}({3:})-tlapse {4:.4f}s-loss{5:.4f}-{6:.4f}-{7:.4f}-{8:.4f}'.
                      format(ep + 1, epochs, i, i_, time.time() - since, loss_, cont_loss_, style_loss_, regularizer_),
                      end='\r', flush=True)
            writer.add_summary(summ_, global_step=i_ + i)
            i += 1

        writer.add_graph(sess.graph)

        print('Saving file ... to fol {{{}}}'.format(self.savepath))
        since = time.time()
        i_ = 0
        for ep in range(epochs):
            i = 0

            optim.minimize(sess, loss_callback=loss_tracking, fetches=[loss, cnt_l, stl_l, regu, summ])
            i_ = i
            audio = sess.run(self.graph['quantized_input'])
            audio = use.inv_mu_law_numpy(audio)
            audio = audio[0,self.late:-self.late]
            #print('audio shape {}'.format(audio.shape))

            # audio_test = sess.run(a)

            sp = os.path.join(self.savepath, 'ep-{}.wav'.format(ep))
            librosa.output.write_wav(sp, audio / np.max(audio), sr=self.sr)
            # sp = os.path.join(self.savepath, 'ep-test-{}.wav'.format(ep))
            # librosa.output.write_wav(sp, audio_test / np.mean(audio_test), sr=self.sr)
            if (ep + 1) % 1 == 0 or i_ < 50:
                gram = sess.run(self.embeds_s1)
                use.show_gram(gram, ep + 1, self.figdir)
                spectrogram.plotstft(sp, plotpath=os.path.join(self.figdir, 'ep_{}_spectro.png'.format(ep+1)))
            if i_ < 50:
                break

    def run(self, cont_file, source, target, epochs, lambd=0.1, gamma=0.1, audio_channel=0, start=1.0):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            phi_t = self.get_style_phi(sess, target)
            phi_s = self.get_style_phi(sess, source)
            aud, _ = use.load_audio(cont_file, sr=self.sr, audio_channel=audio_channel)
            st = int(start * self.sr - self.late)
            aud = aud[st: st + self.batch_size]
            savep = os.path.join(self.savepath, 'ori.wav')
            librosa.output.write_wav(savep, aud[self.late:-self.late], sr=self.sr)
            spectrogram.plotstft(savep, plotpath=os.path.join(self.figdir, 'ori-spec.png'))

            phi_c = self.get_embeds(sess, aud)
            phi = self.get_embeds(sess, aud, is_content=False)
            use.show_gram(phi[1], ep=0, figdir=self.figdir)

            phi[0] = phi[0] + phi_t[0] - phi_s[0]
            phi[1] = phi[1] + phi_t[1] - phi_s[1]
            phi[0] = sess.run(tf.nn.l2_normalize(phi[0], axis=(1, 2)))
            phi[1] = sess.run(tf.nn.l2_normalize(phi[1], axis=(1, 2)))

            self.l_bfgs(sess, phi_c, phi, epochs=epochs, lambd=lambd, gamma=gamma)
            audio = sess.run(self.graph['quantized_input'])
            audio = use.inv_mu_law_numpy(audio)
        return audio[0]


def get_dir(dir, args):
    return use.gt_s_path(use.crt_t_fol(dir), 'oldnew', **vars(args))


def get_fpath(fn, args):
    return os.path.join(args.dir, fn) + '.wav'


def piece_work(args):
    savepath, figdir, logdir = map(lambda dir: get_dir(dir, args),
                                   [args.outdir, args.figdir, args.logdir])

    content, style = map(lambda name: get_fpath(name, args), [args.cont_fn, args.style_fn])

    test = GatysNet(savepath, args.ckpt_path, logdir, figdir, args.stack, args.batch_size, args.sr, args.cont_lyrs,
                    args.channels, args.cnt_channels)
    return test.run(content, content, style, epochs=args.epochs, lambd=args.lambd, gamma=args.gamma, start=args.start)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cont_fn')
    parser.add_argument('style_fn')
    parser.add_argument('--epochs', nargs='?', type=int, default=100)
    parser.add_argument('--batch_size', nargs='?', type=int, default=16384)
    parser.add_argument('--sr', nargs='?', type=int, default=16000)
    parser.add_argument('--stack', nargs='?', type=int, default=None)
    parser.add_argument('--cont_lyrs', nargs='*', type=int, default=[29])
    parser.add_argument('--lambd', nargs='?', type=float, default=0.1)
    parser.add_argument('--gamma', nargs='?', type=float, default=0.00)
    parser.add_argument('--channels', nargs='?', type=int, default=128)
    parser.add_argument('--cnt_channels', nargs='?', type=int, default=128)
    parser.add_argument('--start', nargs='?', type=float, default=1.0)
    #parser.add_argument('--duration', nargs='?', type=int, default=1)

    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--dir', nargs='?', default='./data/src')
    parser.add_argument('--outdir', nargs='?', default='./data/out')
    parser.add_argument('--logdir', nargs='?', default='./log')
    parser.add_argument('--cmt')

    args = parser.parse_args()

    piece_work(args)


if __name__ == '__main__':
    main()