import tensorflow as tf
import numpy.linalg as LA
from nsynth.wavenet.model import Config
from nsynth import utils
from myheap import MyHeap
import numpy as np

def dist(a, b):
    return LA.norm(a-b)

def decode(serialized_example):
    ex = tf.parse_single_example(
        serialized_example,
        features={
            "note_str": tf.FixedLenFeature([], dtype=tf.string),
            "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
            "velocity": tf.FixedLenFeature([1], dtype=tf.int64),
            "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
            "qualities": tf.FixedLenFeature([10], dtype=tf.int64),
            "instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
            "instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
        }
    )
    return ex

def lat_repres(sess, config, graph, layers, wav_tensor=None, wav_path=None, sample_length=64000, sampling_rate=16000):
    if wav_tensor is not None:
        wav = sess.run(wav_tensor)
    else:
        wav = utils.load_audio(wav_path, sample_length, sampling_rate)
    wav = np.reshape(wav, [1, sample_length])


    activations = np.asarray(sess.run([config.extracts[i] for i in layers],
                           feed_dict={graph['X']: wav}))

    return activations

def read_data(data_path, sess, config, graph, layers, wav_rep, k, sample_length, sampling_rate):
    heap = MyHeap(k)

    dataset = tf.data.TFRecordDataset([data_path]).map(decode)
    iterator = dataset.make_one_shot_iterator()
    ex = iterator.get_next()

    try:
        while True:
            rep = lat_repres(sess, config, graph, layers, ex['audio'], None,
                                                  sample_length, sampling_rate)
            heap.push((-dist(wav_rep, rep), rep))
            print(len(heap))
    except tf.errors.OutOfRangeError:
        # Raised when we reach the end of the file.
        pass

def knn(tf_path, file_path, checkpoint_path, layers, k, sample_length=64000, sampling_rate=16000):
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as sess:
        config = Config()
        with tf.device("/gpu:0"):
            x = tf.placeholder(tf.float32, shape=[1, sample_length])
            graph = config.build({'wav': x}, is_training=False)
            graph.update({"X": x})

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        # important
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        wav_rep = lat_repres(sess, config, graph, layers, wav_tensor=None, wav_path=file_path)

        read_data(tf_path, sess, config, graph, layers, wav_rep, k, sample_length, sampling_rate)

def transform(samples, targets, wav, beta):
    avg_s = np.mean(samples)
    avg_t = np.mean(targets)
    omega = avg_t - avg_s
    return wav + beta*omega

def l_bfgs():
    

if __name__=='__main__':
    tf_path = 'data/nsynth-valid.tfrecord'
    file_path = 'test_data/2.wav'
    checkpoint_path = 'nsynth/model/wavenet-ckpt/model.ckpt-200000'
    layers = (1, 10)
    k = 10
    knn(tf_path, file_path, checkpoint_path, layers, k)





