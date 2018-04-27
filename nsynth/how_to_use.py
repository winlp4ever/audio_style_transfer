import numpy as np
import matplotlib.pyplot as plt
from nsynth import utils
from nsynth.wavenet import fastgen
from IPython.display import Audio

filename = './test_data/borrtex.wav'
sampling_rate = 16000
audio = utils.load_audio(filename, sample_length=40000, sr=sampling_rate)
sample_length = audio.shape[0]
print('{} samples, {} seconds'.format(sample_length, sample_length / float(sampling_rate)))

encoding = fastgen.encode(audio, './model/wavenet-ckpt/model.ckpt-200000', sample_length)

print(encoding.shape)

np.save(filename + '.npy', encoding)


fig, axs = plt.subplots(2, 1, figsize=(10, 5))
axs[0].plot(audio)
axs[0].set_title('Audio Signal')
axs[1].plot(encoding[0])
axs[1].set_title('NSynth Encoding')


fastgen.synthesize(encoding, save_paths='./nsynth/test_data/gen_' + filename,
                   checkpoint_path='./model/wavenet-ckpt/model.ckpt-200000',
                   samples_per_save=sample_length)

sampling_rate = 16000
synthesis = utils.load_audio('gen_' + filename, sample_length=sample_length, sr=sampling_rate)
"""
def load_encoding(fname, sample_length=None, sr=16000, ckpt='model.ckpt-200000'):
    audio = utils.load_audio(fname, sample_length=sample_length, sr=sr)
    encoding = fastgen.encode(audio, ckpt, sample_length)
    return audio, encoding

fname = '213259__maurolupo__girl-sings-laa.wav'
sample_length = 32000
audio, encoding = load_encoding(fname, sample_length)
fastgen.synthesize(
    encoding,
    save_paths=['gen_' + fname],
    samples_per_save=sample_length)
synthesis = utils.load_audio('gen_' + fname,
                             sample_length=sample_length,
                             sr=sr)
"""