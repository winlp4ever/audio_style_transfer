import numpy as np

import scipy.io.wavfile as wav
import glob

if __name__ == '__main__':
    for filename in glob.iglob('./test/src/*.wav'):
        sr, audio= wav.read(filename)
        wav.write(filename, sr, audio[:,0])