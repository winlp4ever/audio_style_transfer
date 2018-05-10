import tensorflow as tf
import numpy as np
from nsynth.wavenet.model import Config
from nsynth import utils
from nsynth.wavenet.fastgen import load_nsynth
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def extract_activations(checkpoint_path, layers, wav_data, sample_length=64000, gpu_no=0):
    
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:

        config = Config()
        with tf.device("/gpu:"+str(gpu_no)):
            x = tf.placeholder(tf.float32, shape=[1, 64000])
            graph = config.build({"wav": x}, is_training=False)
            graph.update({"X": x})

        hop_length = config.ae_hop_length
        wav_data, sample_length = utils.trim_for_encoding(wav_data, sample_length, hop_length)
        wav_data = np.reshape(wav_data, (1, 64000))

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        activations = []
        for i in layers:
            activations.append(sess.run(config.extracts[i], feed_dict={graph["X"]: wav_data}))

        repres = np.vstack(activations)
    return repres

def extr_acts(session=None, fr_tfrecord=False, wav_data=None, sample_length=64000):
    assert session is not None
    if fr_tfrecord==True:
        wav_data=

def decode(serialized_example):
    example = tf.parse_single_example(
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
    return example

def knn(wav_data, data_path, nb_voisins=10):
    dataset = tf.data.TFRecordDataset([data_path]).map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_ex = iterator.get_next()
    S = []
    T = []
    with tf.Session() as sess:
        try:
            while True:
                sess.run(next_ex)



def l_bfgs():
    raise ValueError('to be completed!')

def deepfeetinterp():
    raise ValueError('to be completed!')


if __name__ == '__main__':
    layers = (0, 10, 20)
    checkpoint_path = './nsynth/model/wavenet-ckpt/model.ckpt-200000'

    filename = './test_data/2.wav'
    sampling_rate = 16000
    audio = utils.load_audio(filename, sample_length=64000, sr=sampling_rate)

    rep = extract_activations(checkpoint_path, layers, audio)
    print(rep.shape)