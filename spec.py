from nsynth import utils
import matplotlib.pyplot as plt
from scipy.io import wavfile

file_path = './tmp/save_file.wav'

file_path_ = './tmp/save_file_f_u.wav'

_, a = wavfile.read(file_path)
spec_a = utils.specgram(a)
plt.plot(spec_a)
plt.show()