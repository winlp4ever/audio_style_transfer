import tensorflow as tf
from geter import decode
from nsynth.wavenet.model import Config
from nsynth import utils
import numpy as np
from myheap import MyHeap
from numpy.linalg import norm
import librosa
import os, time, argparse
from spectrogram import plotstft
from rainbowgram import plotcqtgram

class Net(object):
    def __init__(self, fpath, spath, tf_path, checkpoint_path, logdir, length=25600, sr=16000):
        self.data = tf.data.TFRecordDataset([tf_path]).map(decode)
        self.checkpoint_path = checkpoint_path
        self.spath = spath
        self.logdir = logdir
        self.length = length
        self.sr = sr
        self.graph = self.build(fpath, length, sr)

    def build(self, fpath, length, sr):
        wav = utils.load_audio(fpath, length, sr)
        wav = np.reshape(wav, [1, length])

        config = Config()
        with tf.device("/gpu:0"):
            x = tf.Variable(initial_value=wav,
                            trainable=True,
                            name='regenerated_wav')

            graph = config.build({'wav': x}, is_training=True)
            graph.update({'X': x})
        return graph

    def load_model(self, sess):
        variables = tf.global_variables()
        variables.remove(self.graph['X'])

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.checkpoint_path)

    def knear(self, sess, type_s, type_t, k):
        it = self.data.make_one_shot_iterator()
        el = it.get_next()

        N_s, N_t = MyHeap(k), MyHeap(k)

        encodings = sess.run(self.graph['encoding'])

        i = 0
        try:
            while True:
                i += 1
                ins = sess.run(el['instrument_family'])

                if ins == type_s:
                    audio = np.reshape(sess.run(el['audio'][:self.length]), [1, self.length])
                    enc = sess.run(self.graph['encoding'], feed_dict={self.graph['X'] : audio})
                    N_s.push((-norm(enc - encodings), i, enc))
                    print('sources - size {} - iterate {}'.format(len(N_s), i))

                elif ins == type_t:
                    audio = np.reshape(sess.run(el['audio'][:self.length]), [1, self.length])
                    enc = sess.run(self.graph['encoding'], feed_dict={self.graph['X']: audio})
                    N_t.push((-norm(enc - encodings), i, enc))
                    print('targets - size {} - iterate {}'.format(len(N_t), i))
        except tf.errors.OutOfRangeError:
            pass

        sources = [N_s[m][2] for m in range(k)]
        targets = [N_t[m][2] for m in range(k)]

        return encodings, sources, targets

    def l_bfgs(self, sess, encodings, epochs, lambd):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        with tf.name_scope('loss'):
            stft = tf.contrib.signal.stft(self.graph['X'], frame_length=1024, frame_step=512, name='stft')
            power_spec = tf.real(stft * tf.conj(stft))
            tf.summary.histogram('spec', power_spec)
            loss = (1 - lambd) * tf.nn.l2_loss(self.graph['encoding'] - encodings) + \
                   lambd * tf.reduce_mean(power_spec)
            tf.summary.scalar('loss', loss)

        summ = tf.summary.merge_all()

        i = 0
        def loss_tracking(loss_, summ_):
            nonlocal i
            print('current loss {}'.format(loss_))
            writer.add_summary(summ_, global_step=i)
            i += 1


        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            var_list=[self.graph['X']],
            method='L-BFGS-B',
            options={'maxiter': epochs})

        optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, summ])

        audio = sess.run(self.graph['X'])

        librosa.output.write_wav(self.spath, audio.T, sr=self.sr)

    def run(self, type_s, type_t, k=10, epochs=100, lambd=0.1):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.load_model(sess)

            if type_s != type_t:
                encodings, sources, targets = self.knear(sess, type_s, type_t, k)
                encodings += np.mean(targets) - np.mean(sources)
            else:
                encodings = sess.run(self.graph['encoding'])

            self.l_bfgs(sess, encodings, epochs, lambd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='filename to transfer style.')
    parser.add_argument('s', help='source type', type=int)
    parser.add_argument('t', help='target type', type=int)
    parser.add_argument('-e', '--epochs', help='number of epochs', nargs='?', type=int, default=1000)
    parser.add_argument('-l', '--lambd', help='lambda value', nargs='?', type=float, default=0.0001)

    args = parser.parse_args()

    def crt_fol(suppath, hour=False):
        date = time.localtime()
        if hour:
            date_fol = os.path.join(suppath, str(date[0]) + str(date[1]) + str(date[2]) + str(date[3]))
        else:
            date_fol = os.path.join(suppath, str(date[0]) + str(date[1]) + str(date[2]))
        if not os.path.exists(date_fol):
            os.makedirs(date_fol)
        return date_fol

    ins_fam = {'bass': 0,
               'brass': 1,
               'flute': 2,
               'guitar': 3,
               'keyboard': 4,
               'mallet': 5,
               'organ': 6,
               'reed': 7,
               'string': 8,
               'synth_lead': 9,
               'vocal': 10}

    inv_map = {k: v for v, k in ins_fam.items()}

    spath = os.path.join(crt_fol('./test/out/'),
                         inv_map[args.s] + '_to_' + inv_map[args.t] + '_' + args.fname + '.wav'
                         )
    fpath = os.path.join('./test/src/', args.fname + '.wav')
    logdir = crt_fol('./log/', True)
    checkpoint_path = './nsynth/model/wavenet-ckpt/model.ckpt-200000'
    tfpath = './data/nsynth-valid.tfrecord'
    figfol = os.path.join('./test/out/fig', args.fname + '.wav')

    net = Net(fpath, spath, tfpath, checkpoint_path, logdir)
    net.run(args.s, args.t, epochs=args.epochs, lambd=args.lambd)

    plotstft(spath, plotpath=os.path.join(figfol, args.fname + '_spec.png'))
    plotcqtgram(spath, savepath=os.path.join(figfol, args.fname + '_spec.png'))

if __name__ == '__main__':
    main()
