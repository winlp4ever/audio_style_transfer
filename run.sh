#!/usr/bin/env bash
source ~/.bashrc
gpucentos
. /home/wp01/interns/leh_2018/anaconda3/etc/profile.d/conda.sh
source activate /home/wp01/interns/leh_2018/anaconda3/envs/deep_learning
python anet.py 'bass' 0 0 --epochs 300 --layers 17 19
