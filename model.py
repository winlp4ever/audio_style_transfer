from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
from nsynth import reader
from nsynth import utils
from nsynth.wavenet import masked


class Cfg(object):
    """Configuration object that helps manage the graph."""

    def __init__(self, train_path=None):
        self.num_iters = 200000
        self.learning_rate_schedule = {
            0: 2e-4,
            90000: 4e-4 / 3,
            120000: 6e-5,
            150000: 4e-5,
            180000: 2e-5,
            210000: 6e-6,
            240000: 2e-6,
        }
        self.ae_hop_length = 512
        self.ae_bottleneck_width = 16
        self.train_path = train_path

        # information to be extracted
        self.extracts = []

    def get_batch(self, batch_size):
        assert self.train_path is not None
        data_train = reader.NSynthDataset(self.train_path, is_training=True)
        return data_train.get_wavenet_batch(batch_size, length=6144)

    @staticmethod
    def _condition(x, encoding):
        """Condition the input on the encoding.

        Args:
          x: The [mb, length, channels] float tensor input.
          encoding: The [mb, encoding_length, channels] float tensor encoding.

        Returns:
          The output after broadcasting the encoding to x's shape and adding them.
        """
        mb, length, channels = x.get_shape().as_list()
        enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
        assert enc_mb == mb
        assert enc_channels == channels

        encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
        x = tf.reshape(x, [mb, enc_length, -1, channels])
        x += encoding
        x = tf.reshape(x, [mb, length, channels])
        x.set_shape([mb, length, channels])
        return x

    def build(self, quantized_inputs, is_training):
        """Build the graph for this configuration.

        Args:
          quantized_inputs: A dict of inputs. For training, should contain 'wav'.
          is_training: Whether we are training or not. Not used in this config.

        Returns:
          A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
          the 'quantized_input', and whatever metrics we want to track for eval.
        """
        del is_training
        num_stages = 10
        num_layers = 30
        filter_length = 3
        width = 512
        skip_width = 256
        ae_num_stages = 10
        ae_num_layers = 30
        ae_filter_length = 3
        ae_width = 128

        # Encode the source with 8-bit Mu-Law.
        x_quantized = quantized_inputs['quantized_wav']
        #x_quantized = utils.mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
        x_scaled = tf.expand_dims(x_scaled, 2)

        ###
        # The Non-Causal Temporal Encoder.
        ###
        enc = masked.conv1d(
            x_scaled,
            causal=False,
            num_filters=ae_width,
            filter_length=ae_filter_length,
            name='ae_startconv')

        with tf.name_scope('encoder'):
            for num_layer in range(ae_num_layers):
                with tf.name_scope('enc-block-{}'.format(num_layer)):
                    dilation = 2 ** (num_layer % ae_num_stages)
                    d_enc = tf.nn.relu(enc)
                    d_enc = masked.conv1d(
                        d_enc,
                        causal=False,
                        num_filters=ae_width,
                        filter_length=ae_filter_length,
                        dilation=dilation,
                        name='ae_dilatedconv_%d' % (num_layer + 1))
                    d_enc = tf.nn.relu(d_enc)

                    d_enc = masked.conv1d(
                        d_enc,
                        num_filters=ae_width,
                        filter_length=1,
                        name='ae_res_%d' % (num_layer + 1))
                    enc += d_enc

                    self.extracts.append(enc)

        enc_ = enc
        self.extracts.append(enc_)

        enc = masked.conv1d(
            enc,
            num_filters=self.ae_bottleneck_width,
            filter_length=1,
            name='ae_bottleneck')

        self.extracts.append(enc)
        enc = masked.pool1d(enc, self.ae_hop_length, name='ae_pool', mode='avg')
        encoding = enc

        print('encoding shape {}'.format(tf.shape(encoding)))

        ###
        # The WaveNet Decoder.
        ###
        l = masked.shift_right(x_scaled)
        l = masked.conv1d(
            l, num_filters=width, filter_length=filter_length, name='startconv')

        # Set up skip connections.
        s = masked.conv1d(
            l, num_filters=skip_width, filter_length=1, name='skip_start')

        # Residual blocks with skip connections.
        with tf.name_scope('decoder'):
            for i in range(num_layers):
                with tf.name_scope('dec-block-{}'.format(i)):
                    dilation = 2 ** (i % num_stages)
                    d_enc = masked.conv1d(
                        l,
                        num_filters=2 * width,
                        filter_length=filter_length,
                        dilation=dilation,
                        name='dilatedconv_%d' % (i + 1))
                    d_enc = self._condition(d_enc,
                                    masked.conv1d(
                                        enc,
                                        num_filters=2 * width,
                                        filter_length=1,
                                        name='cond_map_%d' % (i + 1)))

                    assert d_enc.get_shape().as_list()[2] % 2 == 0
                    m = d_enc.get_shape().as_list()[2] // 2
                    d_sigmoid = tf.sigmoid(d_enc[:, :, :m])
                    d_tanh = tf.tanh(d_enc[:, :, m:])
                    d_enc = d_sigmoid * d_tanh

                    l += masked.conv1d(
                        d_enc, num_filters=width, filter_length=1, name='res_%d' % (i + 1))
                    s += masked.conv1d(
                        d_enc, num_filters=skip_width, filter_length=1, name='skip_%d' % (i + 1))

        s = tf.nn.relu(s)
        s = masked.conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
        s = self._condition(s,
                            masked.conv1d(
                                enc,
                                num_filters=skip_width,
                                filter_length=1,
                                name='cond_map_out1'))
        s = tf.nn.relu(s)

        ###
        # Compute the logits and get the loss.
        ###
        logits = masked.conv1d(s, num_filters=256, filter_length=1, name='logits')
        logits = tf.reshape(logits, [-1, 256])
        probs = tf.nn.softmax(logits, name='softmax')
        x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=x_indices, name='nll'),
            0,
            name='loss')

        return {
            'predictions': probs,
            'loss': loss,
            'eval': {
                'nll': loss
            },
            'quantized_input': x_quantized,
            'encoding': encoding,
            'before_enc': enc_
        }