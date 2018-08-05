import glob
import spectrogram
import os

figdir = './data/fig/84'

for dir in glob.glob('./data/out/84/**'):
    base = os.path.basename(dir)
    if 'gatys' in base:
        print(dir)
        for f in glob.glob(dir + '/**'):
            if f.endswith('ori.wav'):
                fn = os.path.basename(f)
                spectrogram.plotstft(f, plotpath=os.path.join(figdir, os.path.join(base, fn + '.wav')))