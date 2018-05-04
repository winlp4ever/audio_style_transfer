import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from nsynth.wavenet.model import Config
from nsynth import utils
from nsynth.reader import NSynthDataset
from nsynth.wavenet.fastgen import load_nsynth
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def extract_layers(checkpoint_path, wav_data, batch_size=1, sample_length=64000):

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        config = Config()
        with tf.device("/gpu:0"):
            x = tf.placeholder(tf.float32, shape=[1, 64000])
            graph = config.build({"wav": x}, is_training=False)
            graph.update({"X": x})

        hop_length = config.ae_hop_length
        wav_data, sample_length = utils.trim_for_encoding(wav_data, sample_length, hop_length)
        wav_data = np.reshape(wav_data, (1, 64000))
        print(wav_data.shape)

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        writer = tf.summary.FileWriter("output", sess.graph)
        writer.close()
        for i in sess.graph.get_operations():
            print(i.values())

        layer = sess.graph.get_tensor_by_name('dilatedconv_29/W:0')
        print(layer.eval())
        print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False,
                                         all_tensor_names=True)

        length = len(config.extracts)

        encodings = sess.run(config.extracts[length - 1], feed_dict={graph["X"]: wav_data})
        print(type(encodings))
        return encodings


def read_tfrecords(filename, batch_size=1):
    filename_queue = tf.train.string_input_producer([filename],
                                             num_epochs=1, 
                                             )

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    example = tf.parse_single_example(serialized_example,
                                      features={
                                          "note_str": tf.FixedLenFeature([], dtype=tf.string),
                                          "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
                                          "velocity": tf.FixedLenFeature([1], dtype=tf.int64),
                                          "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
                                          "qualities": tf.FixedLenFeature([10], dtype=tf.int64),
                                          "instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
                                          "instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
                                      })
    audio = example['audio']
    audio = tf.reshape(audio, [1, 64000])

    audios = tf.train.shuffle_batch(example['audio'],
                                    batch_size=batch_size,
                                    capacity=3,
                                    num_threads=1,
                                    min_after_dequeue=1)
    return audios


def test_extract():
    filename = './test_data/2.wav'
    sampling_rate = 16000
    audio = utils.load_audio(filename, sample_length=64000, sr=sampling_rate)

    encodings = extract_layers('./nsynth/model/wavenet-ckpt/model.ckpt-200000', audio)
    print(encodings.shape)


def test_read_tfrecords():
    data_path = './data/nsynth-valid.tfrecord'
    # read_tfrecords(data_path)
    with tf.Session() as sess:
        wav = read_tfrecords(data_path)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        sess.run(wav)


if __name__ == '__main__':
    # test_extract()
    test_read_tfrecords()
