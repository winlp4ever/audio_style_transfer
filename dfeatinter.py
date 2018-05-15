import tensorflow as tf
import numpy as np
from nsynth.wavenet.model import Config
from nsynth import utils
from nsynth.wavenet.fastgen import load_nsynth
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from myheap import MyHeap
from geter import decode
import numpy.linalg as LA
from itertools import count

# a global



class DeepFeatInterp():
    def __init__(self, ref_datapath, model_path, layers, sample_length=64000, sampling_rate=16000):

        config = Config()
        with tf.device("/gpu:0"):
            x = tf.Variable(tf.random_normal(shape=[1, 64000], stddev=128), name='regenerated_wav')
            self.graph = config.build({'wav': x}, is_training=False)
            self.graph.update({'X': x})

        self.lat_repr_tens = tf.concat([config.extracts[i] for i in layers], axis=0)

        self.ref_datapath = ref_datapath
        self.model_path = model_path

        self.sample_length = sample_length
        self.sampling_rate = sampling_rate

    def init_reload(self, sess):
        variables = tf.global_variables()
        v = [var for var in variables if var.op.name=='regenerated_wav'][0]
        variables.remove(v)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, self.model_path)

    def load_wav(self, sess, file_path):
        wav = utils.load_audio(file_path, self.sample_length, self.sampling_rate)
        wav = np.reshape(wav, [1, 64000])
        acts = sess.run(self.lat_repr_tens, feed_dict={self.graph['X'] : wav})
        return wav, acts

    def knn(self, sess, file_path, type_1, type_2, k):
        dataset = tf.data.TFRecordDataset([self.ref_datapath]).map(decode)
        iterator = dataset.make_one_shot_iterator()
        ex = iterator.get_next()

        heap_1 = MyHeap(k)
        heap_2 = MyHeap(k)

        _, rep = self.load_wav(sess, file_path)


        try:
            i = 0
            while True:
                i+=1
                type_inst = sess.run(ex['instrument_source'])

                if type_inst == type_1:
                    content = np.reshape(sess.run(ex['audio']), [1, 64000])
                    ex_rep = sess.run(self.lat_repr_tens, feed_dict={self.graph['X'] : content})
                    heap_1.push((-LA.norm(rep - ex_rep), i, ex_rep))
                    print('heap 1 - len {} - iterate {}'.format(len(heap_1), i))

                elif type_inst == type_2:
                    content = np.reshape(sess.run(ex['audio']), [1, 64000])
                    ex_rep = sess.run(self.lat_repr_tens, feed_dict={self.graph['X'] : content})
                    heap_2.push((-LA.norm(rep - ex_rep), i, ex_rep))
                    print('heap 2 - len {} - iterate {}'.format(len(heap_2), i))

        except tf.errors.OutOfRangeError:
            pass
        samples = [heap_1[i][2] for i in range(len(heap_1))]
        targets = [heap_2[i][2] for i in range(len(heap_2))]
        return rep, samples, targets

    @staticmethod
    def transform(rep, heap_1, heap_2, alpha=1):
        return rep + alpha * (np.mean(heap_2, axis=0) - np.mean(heap_1, axis=0))

    def regenerate(self, sess, transform, nb_iter):
        loss = tf.nn.l2_loss(transform - self.lat_repr_tens)
        print(type(self.graph['X']))

        train = tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            var_list= [self.graph['X']],
            method='L-BFGS-B',
            options={'maxiter': nb_iter})
        train.minimize(sess)
        return sess.run([loss, self.graph['X']])

    def run(self, file_path, type_1, type_2, k, nb_iter=100):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            self.init_reload(sess)

            acts, samples, targets = self.knn(sess, file_path, type_1, type_2, k)

            transform = self.transform(acts, samples, targets)

            regen = self.regenerate(sess, transform, nb_iter)
            print('ok')
        return regen


if __name__=='__main__':
    tf_path = './data/nsynth-valid.tfrecord'
    file_path = './nsynth/test_data/borrtex.wav'
    checkpoint_path = './nsynth/model/wavenet-ckpt/model.ckpt-200000'
    layers = (9, 19, 29)
    k = 10
    deepfeat = DeepFeatInterp(tf_path, checkpoint_path, layers)
    loss, regen = deepfeat.run(file_path, 0, 1, k=100, nb_iter=1000)
    print(loss)