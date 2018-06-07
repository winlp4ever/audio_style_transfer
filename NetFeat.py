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
from mdl import Cfg
import random

def mu_law_numpy(x, mu=255):
    out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    out = np.floor(out * 128)
    return out

class Net(object):
    def __init__(self, fpath, spath, tf_path, checkpoint_path, logdir, layers, length=25600, sr=16000):
        self.data = tf.data.TFRecordDataset([tf_path]).map(decode)
        self.checkpoint_path = checkpoint_path
        self.spath = spath
        self.logdir = logdir
        self.length = length
        self.sr = sr
        self.nb_layers = len(layers)
        self.wav, self.graph, self.layers = self.build(fpath, layers, length, sr)

    def build(self, fpath, layers, length, sr):
        #random.seed()
        wav = utils.load_audio(fpath, length, sr)
        #print('max : {}'.format(np.max(wav)))
        #wav[4000:8000,] = librosa.effects.pitch_shift(wav[4000:8000,], sr, random.uniform(-0.5, 0.5))

        #wav += np.random.uniform(-0.04, 0.04, (length))
        wav = np.reshape(wav, [1, length])

        #wav_ = utils.load_audio(fpath, length, sr)
        #wav_ = np.reshape(wav_, [1, length])

        config = Cfg()
        with tf.device("/gpu:0"):
            x = tf.Variable(initial_value=np.zeros([1, length]),
                            trainable=True,
                            name='regenerated_wav')

            graph = config.build({'quantized_wav': x}, is_training=True)
            #graph.update({'X': x})

        lyrs = [config.extracts[i] for i in layers]
        return wav, graph, lyrs

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
                        self.graph['quantized_input'] : mu_law_numpy(audio)})
                    dist = np.sum([norm(encodings[i] - enc[i]) for i in range(self.nb_layers)])

                    N_s.push((-dist, i, enc))
                    print('sources - size {} - iterate {}'.format(len(N_s), i))

                elif ins == type_t:
                    audio = np.reshape(sess.run(el['audio'][:self.length]), [1, self.length])
                    enc = sess.run([self.layers[i] for i in range(self.nb_layers)], feed_dict={
                        self.graph['quantized_input'] : mu_law_numpy(audio)})
                    dist = np.sum([norm(encodings[i] - enc[i]) for i in range(self.nb_layers)])

                    N_t.push((-dist, i, enc))
                    print('targets - size {} - iterate {}'.format(len(N_t), i))
        except tf.errors.OutOfRangeError:
            pass

        sources = [[N_s[m][2][i] for m in range(len(N_s))] for i in range(self.nb_layers)]
        targets = [[N_t[m][2][i] for m in range(len(N_t))] for i in range(self.nb_layers)]

        for i in range(self.nb_layers):
            encodings[i] += np.mean(targets[i], axis=0) - np.mean(sources[i], axis=0)

        return encodings



    def l_bfgs(self, sess, encodings, epochs, lambd):
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        with tf.name_scope('loss'):
            #stft = tf.contrib.signal.stft(self.graph['quantized_input'], frame_length=1024, frame_step=512, name='stft')
            #power_spec = tf.real(stft * tf.conj(stft))
            #tf.summary.histogram('spec', power_spec)

            loss = (1 - lambd) * tf.nn.l2_loss([(self.layers[i] - encodings[i]) for i in range(self.nb_layers)])
                   #lambd * tf.reduce_mean(power_spec)
            tf.summary.scalar('loss', loss)

        summ = tf.summary.merge_all()

        i = 0
        def loss_tracking(loss_, summ_):
            nonlocal i
            print('step {} - loss {}'.format(i, loss_))
            writer.add_summary(summ_, global_step=i)

            i += 1



        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            var_list=[self.graph['quantized_input']],
            method='L-BFGS-B',
            options={'maxiter': epochs})

        optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, summ])

        '''
        optim = tf.train.AdamOptimizer(learning_rate=1e-6)
        trainop = optim.minimize(loss, var_list=[self.graph['quantized_input']])

        sess.run(tf.variables_initializer(optim.variables()))

        for i in range(epochs):
            _, smm, lss = sess.run([trainop, summ, loss])
            writer.add_summary(smm, global_step=i)
            print('step {} - loss {}'.format(i, lss))
        '''
        audio = sess.run(self.graph['quantized_input'])
        audio = utils.inv_mu_law_numpy(audio)

        librosa.output.write_wav(self.spath, audio.T, sr=self.sr)

    def run(self, type_s, type_t, k=10, epochs=100, lambd=0.1):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='filename to transfer style.')
    parser.add_argument('s', help='source type', type=int)
    parser.add_argument('t', help='target type', type=int)
    parser.add_argument('-k', '--knear', help='nb of nearest neighbors', nargs='?', type=int, default=10)
    parser.add_argument('-e', '--epochs', help='number of epochs', nargs='?', type=int, default=100)
    parser.add_argument('-l', '--lambd', help='lambda value', nargs='?', type=float, default=0.0001)
    parser.add_argument('-c', '--cmt', help='comment', nargs='?', default='')

    args = parser.parse_args()

    def crt_fol(suppath, hour=False):
        date = time.localtime()
        if hour:
            date_fol = os.path.join(suppath, str(date[1]) + str(date[2]) + str(date[3]) + str(date[4]))
        else:
            date_fol = os.path.join(suppath, str(date[1]) + str(date[2]))
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
    sname = inv_map[args.s] + '_to_' + inv_map[args.t] + \
                         '_' + args.fname + '_' + str(args.knear) + '_ '+ str(args.lambd) + '_' + args.cmt
    spath = os.path.join(crt_fol('./test/out/'),
                         sname + '.wav'
                         )
    fpath = os.path.join('./test/src/', args.fname + '.wav')
    logdir = crt_fol('./log/', True)
    checkpoint_path = './nsynth/model/wavenet-ckpt/model.ckpt-200000'
    tfpath = './data/nsynth-valid.tfrecord'
    figfol = crt_fol('./test/out/fig')

    layers = [10, 20, 30]

    net = Net(fpath, spath, tfpath, checkpoint_path, logdir, layers)
    net.run(args.s, args.t, k=args.knear, epochs=args.epochs, lambd=args.lambd)

    plotstft(spath, plotpath=os.path.join(figfol, sname + '_spec.png'))
    plotcqtgram(spath, savepath=os.path.join(figfol, sname + '_cqt.png'))

if __name__ == '__main__':
    main()

