import tensorflow as tf
from nsynth.wavenet.model import Config, FastGenerationConfig
from nsynth.wavenet.fastgen import load_nsynth

def extract_layers(checkpoint_path, chosen_layer, batch_size, sample_length):
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        model = Config()

        net = load_nsynth(batch_size=batch_size, sample_length=sample_length)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

if __name__=='__main__':
    extract_layers('./nsynth/model/wavenet-ckpt/')