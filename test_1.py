from scipy.io import wavfile
from numpy.linalg import norm



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filepath', help='filepath of input signal')
parser.add_argument('-s', '--source', help='type of source instrument', type=int)
parser.add_argument('-t', '--target', help='type of target instrument', type=int)

args = parser.parse_args()

print(args.filepath)