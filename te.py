import glob
import spectrogram
import os

figdir = './data/fig/85'

for dir in glob.glob('./data/out/85/**'):
    base = os.path.basename(dir)
    if 'gatys' in base:
        print(dir)
        for f in glob.glob(dir + '/**'):
            if not f.endswith('ori.wav'):
                fn = os.path.basename(f)
                spectrogram.plotstft(f, plotpath=os.path.join(figdir, os.path.join(base, fn + '.png')))