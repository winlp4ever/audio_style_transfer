import tensorflow as tf
from nsynth.wavenet.model import Config
import numpy as np
from nsynth.reader import NSynthDataset
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

    dataset = NSynthDataset(data_path, False)
    dict = dataset.get_example(batch_size)
    print(dict['audio'].get_shape())
    wav = tf.reshape(dict['audio'], [1, 64000])
    with tf.Session(config=session_config) as sess:
        config = Config()
        with tf.device("/gpu:0"):
            graph = config.build({'wav': wav}, is_training=False)
            graph.update({"X": wav})

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        # important
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # ---

        activations = []
        for i in layers:

            activations.append(sess.run(config.extracts[i]))

        repres = np.vstack(activations)
    return repres

if __name__ == '__main__':
    activs = get_activ_fr_dataset('./data/nsynth-valid.tfrecord',
                         './nsynth/model/wavenet-ckpt/model.ckpt-200000',
                         layers=(9, 19))
    print(activs.shape)