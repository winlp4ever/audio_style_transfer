import scipy.io.wavfile as wav
import librosa
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from nsynth import utils

filepath = './test_data/gen_chad_0.wav'
rate, content = wav.read(filepath)

audio = utils.load_audio(filepath, sample_length=64000)
print(type(audio))
print(np.max(audio))

plt.plot(audio)

#plt.plot(content)
plt.savefig('./tmp/im.png')