#!/usr/bin/env bash
source ~/.gridlance.sh
python -V
python khac.py 'pachelbel' 'orchestra' --epochs 50 --lambd 700 --gamma 100 --cont_lyrs 25 --batch_size 8192 --pieces 4