import argparse
import os
import time
from dfeatinter import DeepFeatInterp
from dfeatembed import DFeat
from spectrogram import plotstft

class DefaultList(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 0:
            values = [9, 19, 24, 29, 30]
        setattr(namespace, self.dest, values)

parser = argparse.ArgumentParser()

# add positional arguments
parser.add_argument('fpath', help='relative .wav file path')
parser.add_argument('src', help='instrument type of source files', type=int)
parser.add_argument('trg', help='instrument type of target files', type=int)
# add optional arguments
parser.add_argument('-y', '--layers',
                    help='whether using only embedding activation or not, '
                                          'if specified, which means No, then should specify a list of layers, '
                                           'otherwise layers are set to default',
                    nargs='*', type=int, action=DefaultList)

parser.add_argument('-l', '--lbfgs',
                    help='whether using lbfgs or not. If yes, the nb of iters'
                                          'should be specified, otherwise it will be 1000 by default',
                    nargs='+', type=float)
parser.add_argument('-u', '--supp', help='a string to help better understand output file')
parser.add_argument('-k', '--knear', help='the number of nearest neighbors, equals 10 by default',
                    nargs='?', default=10, type=int)

args = parser.parse_args()

LOGDIR = './log'
MODEL_PATH = './nsynth/model/wavenet-ckpt/model.ckpt-200000'
DATA_PATH = './data/nsynth-valid.tfrecord'
layers = [9, 19, 24, 29, 30]
SRC = './test/src'
OUT = './test/out'
ins_fam = {'bass' : 0,
           'brass' : 1,
           'flute' : 2,
           'guitar' : 3,
           'keyboard' : 4,
           'mallet' : 5,
           'organ' : 6,
           'reed' : 7,
           'string' : 8,
           'synth_lead' : 9,
           'vocal' : 10}

inv_map = {k : v for v,k in ins_fam.items()}



def crt_time_folder(sup_path):
    date = time.localtime()
    date_fol = os.path.join(sup_path, str(date[0]) + '_' + str(date[1]) + '_' + str(date[2]))
    if not os.path.exists(date_fol):
        os.makedirs(date_fol)
    return date_fol

def crt_sname(src, trg, fname, sup):
    s = inv_map[src]+'_to_'+inv_map[trg]+'__'+fname
    if sup:
        s += '_' + sup
    return s

def main():
    data_fol = crt_time_folder(OUT)
    log_fol = crt_time_folder(LOGDIR)
    fig_fol = crt_time_folder(os.path.join(OUT, 'fig/'))
    sname = crt_sname(args.src, args.trg, args.fpath, args.supp)

    bfgs = False
    if args.layers:
        layers = args.layers
        dfeat = DeepFeatInterp(DATA_PATH, MODEL_PATH,
                               layers=layers,
                               save_path=os.path.join(data_fol, sname + '.wav'),
                               logdir=log_fol)

    else:
        dfeat = DFeat(DATA_PATH, MODEL_PATH,
                      save_path=os.path.join(data_fol, sname + '.wav'),
                      logdir=log_fol)

    nb_iter = 0
    lambd = 0.1
    if args.lbfgs:
        assert len(args.lbfgs) == 2
        bfgs = True
        nb_iter = int(args.lbfgs[0])
        lambd = args.lbfgs[1]
    dfeat.run(os.path.join(SRC, args.fpath + '.wav'), args.src, args.trg,
              k=args.knear,
              bfgs=bfgs,
              nb_iter=nb_iter,
              lambd=lambd)

    plotstft(os.path.join(data_fol, sname + '.wav'), plotpath=os.path.join(fig_fol, sname + '.png'))

if __name__ == '__main__':
    main()



