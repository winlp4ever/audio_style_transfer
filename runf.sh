#!/usr/bin/env bash
for s in man voc crickets cat-milk voc_mono; do
    python khac.py 'female' $s --epochs 50 --cont_lyrs 24 --stack 0 --lambd 400 --gamma 100 --batch_size 8192 --pieces 4
done