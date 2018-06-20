import numpy as np

import scipy.io.wavfile as wav
import glob
import ntpath

if __name__ == '__main__':
    for filename in glob.iglob('./data/**'):
        print(ntpath.basename(filename))
print('{{{}}}'.format(2))