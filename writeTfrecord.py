import tensorflow as tf
import librosa

import glob
import ntpath
for dir in glob.iglob('./data/aac/**'):
    id = int(ntpath.basename(dir)[2:])
    print(id)