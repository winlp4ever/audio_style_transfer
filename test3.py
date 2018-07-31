import librosa
import numpy as np
import os
import glob
import spectrogram
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--secs', nargs='?', default=2, type=int)

args = parser.parse_args()

dir = os.path.join('./data/results', 'src_{}secs'.format(args.secs))
figdir = os.path.join(dir, 'specs')
if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(figdir)

for fpath in glob.glob('./data/src/**'):
    if os.path.isdir(fpath):
        continue
    aud, sr = librosa.load(fpath, sr=16000)
    aud = aud[sr: sr + args.secs * sr]
    fname = os.path.basename(fpath)
    spath = os.path.join(dir, fname)
    librosa.output.write_wav(os.path.join(dir, fname), aud, sr)
    rfn = fname.split('.')[0]
    spectrogram.plotstft(spath, plotpath=os.path.join(figdir, rfn + '.png'))
