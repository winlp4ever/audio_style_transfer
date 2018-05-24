import tensorflow as tf
import numpy as np
from nsynth.wavenet.model import Config
from nsynth import utils
import matplotlib.pyplot as plt
from nsynth.wavenet import fastgen
from myheap import MyHeap
from geter import decode
import numpy.linalg as la


class DFeat(object):
    def __init__(self, ref_datapath, model_path, sample_length=64000, sampling_rate=16000, save_path=None):
        assert save_path is not None
        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.save_path = save_path

        config = Config()
        with tf.device("/gpu:0"):
            x = tf.Variable(tf.random_normal(shape=[1, self.sample_length], stddev=1.0),
                            trainable=True,
                            name='regenerated_wav')
            self.graph = config.build({'wav': x}, is_training=False)
            self.graph.update({'X': x})

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
        return encodings

    def knn(self, sess, filepath, type_s, type_t, k):
        dataset = tf.data.TFRecordDataset([self.ref_datapath]).map(decode)
        iterator = dataset.make_one_shot_iterator()
        ex = iterator.get_next()

        heap_s = MyHeap(k)
        heap_t = MyHeap(k)

        encodings = self.get_encodings(sess, filepath)
        i = 0

        try:
            while True:
                i += 1
                type_inst = sess.run(ex['instrument_family'])

                if type_inst == type_s:
                    content = np.reshape(sess.run(ex['audio']), [1, self.sample_length])
                    ex_enc = sess.run(self.graph['encoding'], feed_dict={self.graph['X']: content})

                    heap_s.push((-la.norm(encodings - ex_enc), i, ex_enc))
                    print('sources - shape {} - iterate {}'.format(len(heap_s), i))


                elif type_inst == type_t:
                    content = np.reshape(sess.run(ex['audio']), [1, self.sample_length])
                    ex_enc = sess.run(self.graph['encoding'], feed_dict={self.graph['X']: content})

                    heap_t.push((-la.norm(encodings - ex_enc), i, ex_enc))
                    print('targets - shape {} - iterate {}'.format(len(heap_t), i))

        except tf.errors.OutOfRangeError:
            pass

        sources = [heap_s[m][2] for m in range(k)]
        targets = [heap_t[m][2] for m in range(k)]
        return encodings, sources, targets

    @staticmethod
    def transform(encodings, sources, targets, alpha):
        return encodings + alpha * (np.mean(targets, axis=0) - np.mean(sources, axis=0))

    def regenerate(self, encodings):
        '''
        Regenerate using auto-encoder
        :param sess:
        :param wav:
        :param transform:
        :return:
        '''
        fastgen.synthesize(encodings, [self.save_path], self.model_path)

    def run(self, filepath, type_s, type_t, k):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            self.init_reload(sess)

            encodings, sources, targets = self.knn(sess, filepath, type_s, type_t, k)

            transform = self.transform(encodings, sources, targets, alpha=0.0)

            self.regenerate(transform)

if __name__=='__main__':
    tf_path = './data/nsynth-valid.tfrecord'
    file_path = './test_data/pap/flute.wav'
    checkpoint_path = './nsynth/model/wavenet-ckpt/model.ckpt-200000'
    save_path = './tmp/flute_bass.wav'
    layers = [30]
    deepfeat = DFeat(tf_path, checkpoint_path, save_path=save_path)
    deepfeat.run(file_path, type_s=2, type_t=0, k=10)