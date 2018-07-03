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
from nsynth.wavenet import fastgen


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
        self.graph = self.build(batch_size)

    @staticmethod
    def build(length):
        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(
                initial_value=(np.zeros([1, length])),
                trainable=True,
                name='regenerated_wav'
            )

            graph = config.build({'quantized_wav': x}, is_training=True)

        return graph

    def load_model(self, sess):
        variables = tf.global_variables()
        variables.remove(self.graph['quantized_input'])

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.checkpoint_path)

    def get_embeds(self, sess, aud):
        if len(aud.shape) == 1:
            aud = aud[: self.batch_size]
            aud = np.reshape(aud, [1, self.batch_size])
        embeds = sess.run(self.graph['encoding'],
                          feed_dict={self.graph['quantized_input']: use.mu_law_numpy(aud)})

        embeds += 15

        return embeds

    def get_phi(self, sess, filename, n_components=20, epochs=1000):
        print('load file ...')
        audio, _ = librosa.load(filename, sr=self.sr)
        I = []
        i = 0
        while i + self.batch_size < min(len(audio), 50 * self.batch_size):

            embeds = self.get_embeds(sess, audio[i: i + self.batch_size])
            I.append(embeds)
            print('I size {}'.format(len(I)), end='\r', flush=True)
            i += self.batch_size

        phi = np.concatenate(I, axis=1)[0].T
        w, h = mynmf(phi, n_components=n_components, epochs=epochs)
        return w

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
            audio = use.inv_mu_law_numpy(audio)

            if not (ep + 1) % 10:
                enc = self.get_embeds(sess, audio)
                use.vis_actis(audio[0], enc, self.figdir, ep, [29])

            sp = os.path.join(self.savepath, 'ep-{}.wav'.format(ep))
            librosa.output.write_wav(sp, audio[0] / np.max(audio[0]), sr=self.sr)

    def run(self, main_file, src_file, trg_file, n_components, sample_length=32000):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            ws = self.get_phi(sess, src_file, n_components=n_components)
            wt = self.get_phi(sess, trg_file, n_components=n_components)
            aud, _ = librosa.load(main_file, sr=self.sr)
            enc = self.get_embeds(sess, aud)
            enc = use.transform(enc, ws, wt, n_components=n_components, figdir=self.figdir)
            #self.l_bfgs(sess, enc, epochs=epochs, lambd=0)
            enc -= 15
        print('Saving file ... to fol {{{}}}'.format(self.savepath))
        fastgen.synthesize(
            enc,
            save_paths=[os.path.join(self.savepath, 'synthesize.wav')],
            checkpoint_path=self.checkpoint_path,
            samples_per_save=sample_length)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument('src_fn')
    parser.add_argument('trg_fn')
    parser.add_argument('--n_components', nargs='?', type=int, default=40)
    parser.add_argument('--batch_size', nargs='?', type=int, default=16384)
    parser.add_argument('--sr', nargs='?', type=int, default=16000)

    parser.add_argument('--ckpt_path', nargs='?', default='./nsynth/model/wavenet-ckpt/model.ckpt-200000')
    parser.add_argument('--figdir', nargs='?', default='./data/fig')
    parser.add_argument('--dir', nargs='?', default='./data/src')
    parser.add_argument('--outdir', nargs='?', default='./data/out')
    parser.add_argument('--logdir', nargs='?', default='./log')
    parser.add_argument('--cmt')


    args = parser.parse_args()

    savepath = use.gt_s_path(use.crt_t_fol(args.outdir), **vars(args))
    figdir = use.gt_s_path(use.crt_t_fol(args.figdir), **vars(args))
    logdir = use.gt_s_path(use.crt_t_fol(args.logdir), **vars(args))

    delta = lambda name: os.path.join(args.dir, name) + '.wav'

    fn = delta(args.filename)
    src_fn = delta(args.src_fn)
    trg_fn = delta(args.trg_fn)

    test = dvd_test(savepath, args.ckpt_path, logdir, figdir, args.batch_size, args.sr)
    test.run(fn, src_fn, trg_fn, args.n_components)

if __name__ == '__main__':
    main()








