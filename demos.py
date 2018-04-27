import tensorflow as tf
from nsynth.wavenet.model import Config
from nsynth.wavenet.fastgen import load_nsynth

def extract_layers(checkpoint_path):
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:

        config = Config()
        with tf.device("/gpu:0"):
            x = tf.placeholder(tf.float32, shape=[1, 64000])
            graph = config.build({"wav": x}, is_training=False)
            graph.update({"X": x})
        # net = load_nsynth(batch_size=batch_size, sample_length=40000)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        writer = tf.summary.FileWriter("output", sess.graph)
        writer.close()
        for i in sess.graph.get_operations():
           print(i.values())

        layer = sess.graph.get_tensor_by_name('dilatedconv_29/W:0')
        print(layer.eval())


if __name__=='__main__':
    extract_layers('./nsynth/model/wavenet-ckpt/model.ckpt-200000')