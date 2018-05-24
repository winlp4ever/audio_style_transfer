from scipy.io import wavfile
from numpy.linalg import norm



_, wav = wavfile.read('./tmp/save_file_f.wav')
wav = wav[:64000,]
print(wav)
_, wav_ = wavfile.read('./tmp/save_file_f_u.wav')
print(norm(wav - wav_))
print(wav_)