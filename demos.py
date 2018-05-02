import tensorflow as tf
import numpy as np
from nsynth.wavenet.model import Config
from nsynth import utils
from nsynth.wavenet.fastgen import load_nsynth
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def extract_layers(checkpoint_path, wav_data, batch_size=1, sample_length=64000):
    print(wav_data.shape)

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
        print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False, all_tensor_names=True)

        length = len(config.extracts)

        encodings = sess.run(config.extracts[length - 1], feed_dict={graph["X"]: wav_data})
        print(type(encodings))
        return encodings

if __name__=='__main__':
    filename = './test_data/2.wav'
    sampling_rate = 16000
    audio = utils.load_audio(filename, sample_length=64000, sr=sampling_rate)

    encodings = extract_layers('./nsynth/model/wavenet-ckpt/model.ckpt-200000', audio)
    print(encodings.shape)