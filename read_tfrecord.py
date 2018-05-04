import tensorflow as tf
from nsynth.wavenet.model import Config
import numpy as np
import glob

def decode(serialized_example):
    features = tf.parse_single_example(
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
    return features['audio']

def get_activ_fr_dataset(data_path, checkpoint_path, layers, batch_size=1):
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    dataset = tf.data.TFRecordDataset([data_path]).map(decode)
    iterator = dataset.make_one_shot_iterator()
    wav = iterator.get_next()

    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        config = Config()
        with tf.device("/gpu:0"):
            x = tf.placeholder(tf.float32, shape=[1, 64000])
            graph = config.build({"wav": x}, is_training=False)
            graph.update({"X": x})

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        audio = sess.run(wav)
        audio = np.reshape(audio, [1, 64000])
        print(audio.shape)

        activations = []
        for i in layers:

            activations.append(sess.run(config.extracts[i], feed_dict={graph["X"]: audio}))

        repres = np.vstack(activations)
    return repres


def test_0():
    filename = ['./data/nsynth-valid.tfrecord']
    dataset = tf.data.TFRecordDataset(filename).map(decode)
    iterator = dataset.make_one_shot_iterator()
    wav = iterator.get_next()

    print(type(wav))
    with tf.Session() as sess:
        audio = sess.run(wav)['audio']
        print(audio.shape)

if __name__ == '__main__':
    get_activ_fr_dataset('./data/nsynth-valid.tfrecord',
                         './nsynth/model/wavenet-ckpt/model.ckpt-200000',
                         layers=(9, 19))

'''
    reader = tf.TFRecordReader()
    filenames = glob.glob('./data/nsynth-valid.tfrecord')
    filename_queue = tf.train.string_input_producer(
        filenames)
    _, serialized_example = reader.read(filename_queue)
    feature_set = {'audio': tf.FixedLenFeature([64000], tf.float32)}

    features = tf.parse_single_example(serialized_example, features=feature_set)
    wav = features['audio']

    with tf.Session() as sess:
        print(sess.run(wav))
'''