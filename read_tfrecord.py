import tensorflow as tf
from nsynth.wavenet.model import Config
import numpy as np
from nsynth.reader import NSynthDataset
import glob

def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            "id": tf.FixedLenFeature([], dtype=tf.int64),
            "audio": tf.FixedLenFeature([], dtype=tf.float32)
        }
    )
    return features




if __name__ == '__main__':
    print('to do something!')