import tensorflow as tf
import numpy as np
from nsynth.wavenet.model import Config
from nsynth import utils
import matplotlib.pyplot as plt
from nsynth.wavenet import fastgen
import librosa

plt.switch_backend('agg')

from nsynth.wavenet.fastgen import load_nsynth
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from myheap import MyHeap
from geter import decode
import numpy.linalg as LA


class DeepFeatInterp():
    def __init__(self, ref_datapath, model_path, layers, sample_length=64000, sampling_rate=16000, save_path=None):
        assert save_path is not None
        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.save_path = save_path

        self.nb_layers = len(layers)

        config = Config()
        with tf.device("/gpu:0"):
            x = tf.Variable(tf.random_normal(shape=[1, self.sample_length], stddev=1.0),
                            trainable=True,
                            name='regenerated_wav')
            self.graph = config.build({'wav': x}, is_training=False)
            self.graph.update({'X': x})

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

        heap_s, heap_t = MyHeap(k), MyHeap(k)

        wav, rep = self.load_wav(sess, file_path)

        try:
            i = 0
            while True:
                i += 1
                type_inst = sess.run(ex['instrument_family'])

                if type_inst == type_s:
                    content = np.reshape(sess.run(ex['audio']), [1, self.sample_length])
                    ex_rep = sess.run(self.lat_repr_tens, feed_dict={self.graph['X']: content})
                    heap_s.push((-LA.norm(rep - ex_rep), i, ex_rep))
                    print('samples - len {} - iterate {}'.format(len(heap_s), i))

                elif type_inst == type_t:
                    content = np.reshape(sess.run(ex['audio']), [1, self.sample_length])
                    ex_rep = sess.run(self.lat_repr_tens, feed_dict={self.graph['X']: content})
                    heap_t.push((-LA.norm(rep - ex_rep), i, ex_rep))
                    print('targets - len {} - iterate {}'.format(len(heap_t), i))

        except tf.errors.OutOfRangeError:
            pass
        samples = [heap_s[i][2] for i in range(len(heap_s))]
        targets = [heap_t[i][2] for i in range(len(heap_t))]
        return wav, rep, samples, targets

    @staticmethod
    def transform(rep, samples, targets, alpha=1):
        return rep + alpha * (np.mean(targets, axis=0) - np.mean(samples, axis=0))

    def regen_opt(self, sess, transform, nb_iter):
        '''
        Regenerate using optimization
        :param sess:
        :param transform:
        :param nb_iter:
        :return:
        '''
        loss = tf.nn.l2_loss(transform - self.lat_repr_tens)
        print(type(self.graph['X']))

        train = tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            var_list=[self.graph['X']],
            method='L-BFGS-B',
            options={'maxiter': nb_iter})
        train.minimize(sess)

        audio = sess.run(self.graph['X'])
        librosa.output.write_wav(self.save_path, audio, sr=self.sampling_rate)

        print(sess.run(loss))

    def regen_aut(self, sess, wav, transform):
        '''
        Regenerate using auto-encoder
        :param sess:
        :param wav:
        :param transform:
        :return:
        '''
        lays = self.activ_layers
        lays.append(self.graph['X'])

        transform_ = np.expand_dims(transform, axis=1)
        values = [transform_[i] for i in range(self.nb_layers)]
        values.append(wav)

        encodings = sess.run(self.graph['encoding'],
                             feed_dict={lays[i] : values[i] for i in range(self.nb_layers+1)
                                        })
        fastgen.synthesize(encodings, [self.save_path], self.model_path)

    def run(self, file_path, type_s, type_t, k, nb_iter=100, bfgs=True):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            self.init_reload(sess)

            wav, acts, samples, targets = self.knn(sess, file_path, type_s, type_t, k)

            transform = self.transform(acts, samples, targets)

            if bfgs:
                self.regen_opt(sess, transform, nb_iter)
                return

            self.regen_aut(sess, wav, transform)
            return



if __name__ == '__main__':
    tf_path = './data/nsynth-valid.tfrecord'
    file_path = './test_data/gen_chad_0.wav'
    checkpoint_path = './nsynth/model/wavenet-ckpt/model.ckpt-200000'
    save_path = './tmp/save_file.wav'
    layers = (9, 19, 29)
    deepfeat = DeepFeatInterp(tf_path, checkpoint_path, layers, save_path=save_path)
    deepfeat.run(file_path, type_s=4, type_t=3, k=10, nb_iter=int(1e12), bfgs=False)

