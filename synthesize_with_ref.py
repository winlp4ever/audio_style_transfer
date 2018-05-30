import tensorflow as tf
from nsynth.wavenet.model import Config
import numpy as np
from scipy.io import wavfile
from nsynth import utils

def save_wav(batch_audio, batch_save_paths):
    for audio, name in zip(batch_audio, batch_save_paths):
        tf.logging.info("Saving: %s" % name)
        wavfile.write(name, 16000, audio)

def synthesize_with_ref(encodings,
    save_paths,
    checkpoint_path = "model.ckpt-200000",
    samples_per_save = 1000):
    hop_length = Config().ae_hop_length
    # Get lengths
    batch_size = encodings.shape[0]
    encoding_length = encodings.shape[1]
    total_length = encoding_length * hop_length

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        net = load_fastgen_nsynth(batch_size=batch_size)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        # initialize queues w/ 0s
        sess.run(net["init_ops"])

        # Regenerate the audio file sample by sample
        audio_batch = np.zeros(
            (
                batch_size,
                total_length,
            ), dtype=np.float32)
        audio = np.zeros([batch_size, 1])

        for sample_i in range(total_length):
            enc_i = sample_i // hop_length
            pmf = sess.run(
                [net["predictions"], net["push_ops"]],
                feed_dict={
                    net["X"]: audio,
                    net["encoding"]: encodings[:, enc_i, :]
                })[0]
            sample_bin = sample_categorical(pmf)
            audio = utils.inv_mu_law_numpy(sample_bin - 128)
            audio_batch[:, sample_i] = audio[:, 0]
            if sample_i % 100 == 0:
                tf.logging.info("Sample: %d" % sample_i)
                print("Sample: %d" % sample_i)
            if sample_i % samples_per_save == 0:
                save_wav(audio_batch, save_paths)
    save_wav(audio_batch, save_paths)
