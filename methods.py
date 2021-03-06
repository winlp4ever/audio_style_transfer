import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from model import cfg
import utils
import librosa
import time
import argparse
import matplotlib.pyplot as plt
import spectrogram

tf.logging.set_verbosity(tf.logging.WARN)

plt.switch_backend('agg')


class GatysNet(object):
    def __init__(self,
                 savepath='./data/out',
                 checkpoint_path='./nsynth/model/wavenet-ckpt/model.ckpt-200000',
                 logdir='./log',
                 figdir='./data/fig',
                 stack=0,
                 batch_size=16384,
                 sr=16000,
                 cont_lyr_ids=[29],
                 nb_channels=128,
                 cnt_channels=128,
                 gatys=False,
                 style_lyr_ids=None):
        self.logdir = logdir
        self.savepath = savepath
        self.checkpoint_path = checkpoint_path
        self.figdir = figdir
        self.batch_size = batch_size
        self.sr = sr
        self.late = (batch_size - (batch_size // 4096) * 4000) // 2
        self.cont_lyr_ids = cont_lyr_ids
        self.graph, self.embeds_c, self.embeds_s, self.gatys = self.build(batch_size, cont_lyr_ids, stack, nb_channels,
                                                              cnt_channels, gatys, style_lyr_ids)

    @staticmethod
    def build(length, cont_lyr_ids, stack, nb_channels, cnt_channels=128, gatys=False, style_lyr_ids=None):
        tf.reset_default_graph()
        config = cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=np.zeros([1, length]) + 1e-6,
                trainable=True,
                name='regenerated_wav',
                dtype=tf.float32
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

            cont_embeds = tf.concat([config.extracts[i][:, :, :cnt_channels] for i in cont_lyr_ids], axis=2)[0]

            if style_lyr_ids is not None:
                assert isinstance(style_lyr_ids, (tuple, list)), "style_lyr_ids must be of type tuple or list!"
                stl = tf.concat([config.extracts[i] for i in style_lyr_ids], axis=0)
            elif stack is not None:
                stl = tf.concat([config.extracts[i] for i in range(stack * 10, stack * 10 + 10)], axis=0)
            else:
                stl = tf.concat([config.extracts[i] for i in range(30)], axis=0)

            if not gatys:
                stl = tf.transpose(stl, perm=[2, 0, 1])
            else:
                stl = tf.transpose(stl, perm=[0, 2, 1])

            style_embeds = tf.matmul(stl, tf.transpose(stl, perm=[0, 2, 1]))
            style_embeds = tf.nn.l2_normalize(style_embeds, axis=(1, 2))
            if nb_channels < 128 and not gatys:
                style_embeds = style_embeds[:nb_channels]
        return graph, cont_embeds, style_embeds, gatys

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
                        feed_dict={self.graph['quantized_input']: utils.mu_law_numpy(aud)})

    def get_style_phi(self, sess, filename, max_examples=5, show_mat=True):
        print('load file ...')
        audio, _ = utils.load_audio(filename, sr=self.sr, audio_channel=0)
        I = []
        i = 0
        while i + self.batch_size <= min(len(audio), max_examples * self.batch_size):
            embeds = self.get_embeds(sess, audio[i: i + self.batch_size], is_content=False)
            I.append(embeds)
            print('I size {}'.format(len(I)), end='\r', flush=True)
            i += self.batch_size

        phi = np.mean(I, axis=0)
        if show_mat:
            utils.show_gram(phi, figdir=self.figdir, gatys=self.gatys)
        return phi

    def define_loss(self, name, stl_emb, cnt_emb, lambd, gamma, gpu):
        with tf.device(gpu):
            with tf.name_scope(name):
                content_loss = tf.reduce_mean(tf.square(self.embeds_c - cnt_emb))
                content_loss *= 10
                style_loss = tf.reduce_mean(tf.square(self.embeds_s - stl_emb))
                style_loss *= 1e3

                a = utils.inv_mu_law(self.graph['quantized_input'][0])
                regularizer = tf.contrib.signal.stft(a, frame_length=1024, frame_step=512, name='stft')
                regularizer = tf.reduce_mean(utils.abs(tf.real(regularizer)) + utils.abs(tf.imag(regularizer)))

                loss = content_loss + lambd * style_loss + gamma * regularizer

                tf.summary.scalar('content_loss', content_loss)
                tf.summary.scalar('style_loss', style_loss)
                tf.summary.scalar('regularizer', regularizer)
                tf.summary.scalar('main_loss', loss)

            with tf.name_scope('optim'):
                optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    loss,
                    var_list=[self.graph['quantized_input']],
                    method='L-BFGS-B',
                    options={'maxiter': 100})
        return content_loss, style_loss, regularizer, loss, optimizer

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
            audio = utils.inv_mu_law_numpy(audio)
            audio = audio[0,self.late:-self.late]

            sp = os.path.join(self.savepath, 'ep-{}.wav'.format(ep))

            if (ep + 1) % 1 == 0 or i_ < 50:
                librosa.output.write_wav(sp, audio / np.max(audio), sr=self.sr)
                grams = sess.run(self.embeds_s)
                utils.show_gram(grams, ep + 1, self.figdir, gatys=self.gatys)
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
            phi_s = self.get_style_phi(sess, source, show_mat=False)
            aud, _ = utils.load_audio(cont_file, sr=self.sr, audio_channel=audio_channel)
            st = int(start * self.sr - self.late)
            aud = aud[st: st + self.batch_size]
            savep = os.path.join(self.savepath, 'ori.wav')
            librosa.output.write_wav(savep, aud[self.late:-self.late], sr=self.sr)
            spectrogram.plotstft(savep, plotpath=os.path.join(self.figdir, 'ori-spec.png'))

            style_aud, _ = utils.load_audio(target, sr=self.sr, audio_channel=audio_channel)
            style_aud = style_aud[st: st + self.batch_size]
            saves = os.path.join(self.savepath, 'style.wav')
            librosa.output.write_wav(saves, style_aud[self.late: -self.late], sr=self.sr)
            spectrogram.plotstft(saves, plotpath=os.path.join(self.figdir, 'style-spec.png'))

            phi_c = self.get_embeds(sess, aud)
            phi = self.get_embeds(sess, aud, is_content=False)
            utils.show_gram(phi, ep=0, figdir=self.figdir, gatys=self.gatys)

            phi = phi + phi_t - phi_s
            phi = sess.run(tf.nn.l2_normalize(phi, axis=(1, 2)))
            self.l_bfgs(sess, phi_c, phi, epochs=epochs, lambd=lambd, gamma=gamma)
            audio = sess.run(self.graph['quantized_input'])
            audio = utils.inv_mu_law_numpy(audio)
        return audio[0]


def get_dir(dir, args):
    return utils.gt_s_path(utils.crt_t_fol(dir), **vars(args))


def get_fpath(fn, args):
    return os.path.join(args.dir, fn) + '.wav'


def piece_work(args):
    savepath, logdir = map(lambda dir: get_dir(dir, args),
                                   [args.outdir, args.logdir])

    figdir = os.path.join(savepath, 'fig')

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    content, style = map(lambda name: get_fpath(name, args), [args.cont_fn, args.style_fn])

    test = GatysNet(savepath, args.ckpt_path, logdir, figdir, args.stack, args.batch_size, args.sr, args.cont_lyrs,
                    args.channels, args.cnt_channels, args.gatys, args.style_lyrs)
    return test.run(content, content, style, epochs=args.epochs, lambd=args.lambd, gamma=args.gamma, start=args.start)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cont_fn', help='relative content file name')
    parser.add_argument('style_fn', help='relative style file name')
    parser.add_argument('--epochs', help='number of epochs, each epoch contains 100 iterations of optimization',
                        nargs='?', type=int, default=100)
    parser.add_argument('--batch_size', help='length of output signal, must be divided by 4096', nargs='?', type=int, default=16384)
    parser.add_argument('--sr', help='sampling rate, default to 16kHz', nargs='?', type=int, default=16000)
    parser.add_argument('--stack', help='stack of layers chosen for computing style loss. Have effects only if style_lyrs is None. There are 3 stacks, each of 10 layers. If None'
                                        ' then all three stacks will be taken into account', nargs='?', type=int, default=None)
    parser.add_argument('--cont_lyrs', nargs='*', type=int, default=[29])
    parser.add_argument('--style_lyrs', nargs='*', type=int)
    parser.add_argument('--lambd', help='style loss scalar coefficient', nargs='?', type=float, default=100.0)
    parser.add_argument('--gamma', help='regularizer scalar coefficient', nargs='?', type=float, default=0.0)
    parser.add_argument('--channels', help='how many channels taken into account for style loss', nargs='?', type=int, default=128)
    parser.add_argument('--cnt_channels', help='how many channels taken into account for content loss', nargs='?', type=int, default=128)
    parser.add_argument('--start', nargs='?', type=float, default=1.0)
    parser.add_argument('--gatys', nargs='?', type=bool, default=False, const=True)

    parser.add_argument('--ckpt_path', help="path to the pretrained model's checkpoint path", nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    parser.add_argument('--dir', help='path to source files, should be where to store reference style and content files', nargs='?', default='./data/src')
    parser.add_argument('--outdir', help='path to output', nargs='?', default='./data/out')
    parser.add_argument('--logdir', help='path to logs', nargs='?', default='./log')
    parser.add_argument('--cmt')

    args = parser.parse_args()

    piece_work(args)


if __name__ == '__main__':
    main()