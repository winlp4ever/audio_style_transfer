import tensorflow as tf
import numpy as np
from nsynth.wavenet.model import Config
from nsynth import utils
import matplotlib.pyplot as plt
from nsynth.wavenet import fastgen
import librosa
from scipy.io import wavfile
from spectrogram import plotstft
from synthesize_with_ref import synthesize_with_ref

plt.switch_backend('agg')

from nsynth.wavenet.fastgen import load_nsynth
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from myheap import MyHeap
from geter import decode
from numpy.linalg import norm


class DeepFeatInterp():
    def __init__(self, ref_datapath, model_path, layers, sample_length=25600,
                 sampling_rate=16000, save_path=None, logdir=None):
        assert save_path
        assert logdir
        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.save_path = save_path
        self.logdir = logdir

        self.nb_layers = len(layers)

        config = Config()
        with tf.device("/gpu:0"):
            x = tf.Variable(tf.random_normal(shape=[1, self.sample_length], stddev=1.0),
                            trainable=True,
                            name='regenerated_wav')
            self.graph = config.build({'wav': x}, is_training=False)
            self.graph.update({'X': x})

        print('\n len extracts : {} \n last layer shape : {}'.format(len(config.extracts),
                                                                   tf.shape(config.extracts[30])))

        self.activ_layers = [config.extracts[i] for i in layers]
        self.lat_repr_tens = tf.concat(self.activ_layers, axis=0)

        self.ref_datapath = ref_datapath
        self.model_path = model_path

    def init_reload(self, sess):
        variables = tf.global_variables()
        v = [var for var in variables if var.op.name == 'regenerated_wav'][0]
        variables.remove(v)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.model_path)

    def load_wav(self, sess, file_path):
        wav = utils.load_audio(file_path, self.sample_length, self.sampling_rate)
        wav = np.reshape(wav, [1, self.sample_length])
        acts = np.asarray(sess.run(self.lat_repr_tens, feed_dict={self.graph['X']: wav}))
        print('shape acts : {}'.format(acts.shape))
        return wav, acts

    def knn(self, sess, file_path, type_s, type_t, k):
        dataset = tf.data.TFRecordDataset([self.ref_datapath]).map(decode)
        iterator = dataset.make_one_shot_iterator()
        ex = iterator.get_next()

        heap_s = MyHeap(k)
        heap_t = MyHeap(k)

        wav, rep = self.load_wav(sess, file_path)

        try:
            i = 0
            while True:
                i += 1
                type_inst = sess.run(ex['instrument_family'])

                if type_inst == type_s:
                    content = np.reshape(sess.run(ex['audio'])[:self.sample_length], [1, self.sample_length])
                    ex_rep = sess.run(self.lat_repr_tens, feed_dict={self.graph['X']: content})

                    heap_s.push((-norm(rep - ex_rep), i, ex_rep))
                    print('sources - shape {} - iterate {}'.format(len(heap_s), i))


                elif type_inst == type_t:
                    content = np.reshape(sess.run(ex['audio'])[:self.sample_length], [1, self.sample_length])
                    ex_rep = sess.run(self.lat_repr_tens, feed_dict={self.graph['X']: content})

                    heap_t.push((-norm(rep - ex_rep), i, ex_rep))
                    print('targets - shape {} - iterate {}'.format(len(heap_t), i))

        except tf.errors.OutOfRangeError:
            pass

        sources = [heap_s[m][2] for m in range(k)]
        targets = [heap_t[m][2] for m in range(k)]
        return wav, rep, sources, targets

    @staticmethod
    def transform(rep, sources, targets, alpha=1.0):
        return rep + alpha * (np.mean(targets, axis=0) - np.mean(sources, axis=0))

    def get_encodings(self, sess, wav, transform):
        lays = self.activ_layers
        lays.append(self.graph['X'])

        transform_ = np.expand_dims(transform, axis=1)
        values = [transform_[i] for i in range(self.nb_layers)]
        values.append(wav)

        encodings = sess.run(self.graph['encoding'],
                             feed_dict={lays[i]: values[i] for i in range(self.nb_layers + 1)
                                        })
        return encodings

    def regen_opt(self, sess, wav, encodings, nb_iter, lambd):
        '''
        Regenerate using optimization
        :param sess:
        :param encodings:
        :param nb_iter:
        :return:
        '''

        self.graph['X'] = tf.Variable(initial_value=wav, dtype=tf.float32)

        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)

        tf.summary.histogram('input', self.graph['X'])

        with tf.name_scope('loss'):
            stft = tf.contrib.signal.stft(signals=self.graph['X'],
                                          frame_length=1024,
                                          frame_step=512, name='stft')
            power_spec = tf.real(stft * tf.conj(stft))
            tf.summary.histogram('spec', power_spec)
            loss = (1 - lambd) * tf.nn.l2_loss(encodings - self.graph['encoding']) + \
                   lambd * tf.reduce_mean(power_spec)

            tf.summary.scalar('loss', loss)

        summ = tf.summary.merge_all()
        print(type(self.graph['X']))

        step = 0

        def loss_tracking(loss_, summ_):
            nonlocal step
            print('current loss is %s' % loss_)
            writer.add_summary(summ_, global_step=step)
            step += 1

        train = tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            var_list=[self.graph['X']],
            method='L-BFGS-B',
            options={'maxiter': nb_iter})

        sess.run(tf.variables_initializer([self.graph['X']]))
        train.minimize(sess, loss_callback=loss_tracking ,fetches=[loss, summ])

        audio = sess.run(self.graph['X'])

        print(audio)

        #audio = utils.inv_mu_law_numpy(audio - 128)

        wavfile.write(self.save_path, self.sampling_rate, audio.T)

    def regen_aut(self, sess, wav, transform):
        '''
        Regenerate using auto-encoder
        :param sess:
        :param wav:
        :param transform:
        :return:
        '''
        encodings = self.get_encodings(sess, wav, transform)
        synthesize_with_ref(encodings, wav=wav, save_paths=[self.save_path], checkpoint_path=self.model_path)

    def run(self, file_path, type_s, type_t, k, bfgs=False, nb_iter=100, lambd=0.1):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            self.init_reload(sess)
            if type_s != type_t:
                wav, acts, sources, targets = self.knn(sess, file_path, type_s, type_t, k)
                transform = self.transform(acts, sources, targets, alpha=1.0)

            else:
                wav, transform = self.load_wav(sess, file_path)

            if bfgs:
                self.regen_opt(sess, wav, transform, nb_iter, lambd)
                return

            self.regen_aut(sess, wav, transform)
            return
