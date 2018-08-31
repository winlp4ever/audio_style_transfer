import tensorflow as tf
import numpy as np
from nsynth.wavenet.model import Config
from nsynth import utils
import matplotlib.pyplot as plt
from nsynth.wavenet import fastgen
from myheap import MyHeap
from geter import decode
import numpy.linalg as la
import librosa
from synthesize_with_ref import synthesize_with_ref


class DFeat(object):
    def __init__(self, ref_datapath, model_path, sample_length=25600,
                 sampling_rate=16000, save_path=None, logdir=None):
        assert save_path
        assert logdir
        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.save_path = save_path
        self.logdir = logdir

        config = Config()
        with tf.device("/gpu:0"):
            x = tf.Variable(tf.random_normal(shape=[1, self.sample_length], stddev=1.0),
                            trainable=True,
                            name='regenerated_wav')

            self.graph = config.build({'wav': x}, is_training=False)
            self.graph.update({'X': x})

        print('extract len : {}'.format(len(config.extracts)))

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

    def get_encodings(self, sess, filepath):
        wav = utils.load_audio(filepath, self.sample_length, self.sampling_rate)
        wav = np.reshape(wav, [1, self.sample_length])

        encodings = sess.run(self.graph['encoding'], feed_dict={self.graph['X'] : wav})
        return wav, encodings

    def knn(self, sess, filepath, type_s, type_t, k):
        dataset = tf.data.TFRecordDataset([self.ref_datapath]).map(decode)
        iterator = dataset.make_one_shot_iterator()
        ex = iterator.get_next()

        heap_s = MyHeap(k)
        heap_t = MyHeap(k)

        wav, encodings = self.get_encodings(sess, filepath)
        i = 0

        try:
            while True:
                i += 1
                type_inst = sess.run(ex['instrument_family'])

                if type_inst == type_s:
                    content = np.reshape(sess.run(ex['audio'][:self.sample_length]), [1, self.sample_length])
                    ex_enc = sess.run(self.graph['encoding'], feed_dict={self.graph['X']: content})

                    heap_s.push((-la.norm(encodings - ex_enc), i, ex_enc))
                    print('sources - shape {} - iterate {}'.format(len(heap_s), i))


                elif type_inst == type_t:
                    content = np.reshape(sess.run(ex['audio'][:self.sample_length]), [1, self.sample_length])
                    ex_enc = sess.run(self.graph['encoding'], feed_dict={self.graph['X']: content})

                    heap_t.push((-la.norm(encodings - ex_enc), i, ex_enc))
                    print('targets - shape {} - iterate {}'.format(len(heap_t), i))

        except tf.errors.OutOfRangeError:
            pass

        sources = [heap_s[m][2] for m in range(k)]
        targets = [heap_t[m][2] for m in range(k)]
        return wav, encodings, sources, targets

    @staticmethod
    def transform(encodings, sources, targets, alpha):
        return encodings + np.mean(targets, axis=0) - np.mean(sources, axis=0)

    def lbfgs(self, sess, wav, encodings, lambd, nb_iter):

        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_graph(sess.graph)


        var = tf.Variable(initial_value=wav, dtype=tf.float32)

        assign = tf.assign(self.graph['X'], var)
        tf.summary.histogram('input', self.graph['X'])

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
            var_list=[var],
            method='L-BFGS-B',
            options={'maxiter': nb_iter})

        sess.run(tf.variables_initializer([var]))


        optimizer.minimize(sess, loss_callback=loss_tracking, fetches=[loss, summ])

        audio = sess.run(self.graph['X'])

        librosa.output.write_wav(self.save_path, audio.T, sr=self.sampling_rate)

    def regenerate(self, encodings):
        '''
        Regenerate using auto-encoder
        :param sess:
        :param wav:
        :param transform:
        :return:
        '''
        fastgen.synthesize(encodings, [self.save_path], self.model_path)

    def run(self, filepath, type_s, type_t, k, bfgs=False, nb_iter=100, lambd=0.1):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            self.init_reload(sess)

            if type_s != type_t:
                wav, encodings, sources, targets = self.knn(sess, filepath, type_s, type_t, k)
                transform = self.transform(encodings, sources, targets, alpha=1.0)
            else:
                wav, transform = self.get_encodings(sess, filepath)
            if bfgs:
                self.lbfgs(sess, wav, transform, lambd, nb_iter)
                return
            self.regenerate(transform)
            return
