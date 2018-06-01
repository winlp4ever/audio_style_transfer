import os

import librosa
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['svg.fonttype'] = 'none'
import numpy as np
from scipy.io.wavfile import read as readwav

# Constants
n_fft = 512
hop_length = 256
SR = 16000
over_sample = 4
res_factor = 0.8
octaves = 6
notes_per_octave = 10

# Plotting functions
cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),

         'alpha': ((0.0, 1.0, 1.0),
                   (1.0, 0.0, 0.0))
         }

my_mask = matplotlib.colors.LinearSegmentedColormap('MyMask', cdict)
plt.register_cmap(cmap=my_mask)

def rainbowgram(audio,
                sr,
                peak=80.,
                n_fft=512,
                hop_length=None,
                over_sample=4,
                res_factor=0.8,
                octaves=6,
                notes_per_octave=10):
    if not hop_length:
        hop_length = n_fft // 2

    cqt = librosa.cqt(audio, sr=sr, hop_length=hop_length,
                    bins_per_octave=int(notes_per_octave * over_sample),
                    n_bins=int(octaves * notes_per_octave * over_sample),
                    filter_scale=res_factor,
                    fmin=librosa.note_to_hz('C2'))
    mag, phase = librosa.core.magphase(cqt)
    phase_angle = np.angle(phase)

    mag = (librosa.power_to_db(
        mag ** 2, amin=1e-13, top_db=peak, ref=np.max) / peak) + 1
    phase_unwrapped = np.unwrap(phase_angle)
    p = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
    p = np.concatenate([phase_unwrapped[:, 0:1], p], axis=1) / np.pi
    return mag, p

def plotcqtgram(filepath, savepath=None):
    sr, audio = readwav(filepath)
    audio = audio.astype(np.float32)
    mag, p = rainbowgram(audio, sr)
    fig, ax = plt.subplots()

    ax.matshow(p[::-1, :], cmap=plt.cm.rainbow)
    ax.matshow(mag[::-1, :], cmap=my_mask)
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

if __name__ == '__main__':
    path = './test/out/2018531/bass_to_flute__bass_avg_no_embed.wav'
    plotcqtgram(path)