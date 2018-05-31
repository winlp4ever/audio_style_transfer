from nsynth import utils
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


file_path = './test_data/gui.wav'


a = utils.load_audio(file_path, sr=16000)
print(len(a))

b = np.zeros(shape=(64000,))

b[:len(a)] = a

wavfile.write('./test_data/gen_gui.wav', rate=16000, data=b)