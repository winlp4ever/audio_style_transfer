import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import nsynth.utils as utils
import librosa.display
from spectrogram import plotstft

filepath = './test/src/flute.wav'

audio, sr = librosa.load(filepath)
fft_config = dict(
        n_fft=512, win_length=512, hop_length=256, center=True)
spec = librosa.stft(audio, **fft_config)

mag, phase = librosa.core.magphase(spec)
mag = (librosa.power_to_db(
                mag ** 2, amin=1e-13, top_db=120., ref=np.max) / 120.) + 1
mag.astype(np.float32)

print(mag.shape)
librosa.display.specshow(mag, y_axis = 'log', x_axis = 'time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

plotstft(filepath)


