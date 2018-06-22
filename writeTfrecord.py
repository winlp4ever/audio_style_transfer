import tensorflow as tf
import librosa

import glob
import ntpath


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


train_filename = './data/dataset/aac-test.tfrecord'

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

batch_size = 16384

for dir in glob.iglob('./data/aac/**'):
    id = int(ntpath.basename(dir)[2:])
    for path in glob.iglob('{}/**/*wav'.format(dir)):
        print(path)

        aud, _ = librosa.load(path, sr=16000)

        length = len(aud)

        nb_batches = length // batch_size

        for i in range(nb_batches):
            audio = aud[i * batch_size:(i + 1) * batch_size]

            # create a feature
            feature = {'id': _int64_feature(id),
                       'audio': _bytes_feature(tf.compat.as_bytes(audio.tostring()))}

            # create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and wrtie on the file
            writer.write(example.SerializeToString())

writer.close()
