#! /usr/bin/env bash
for c in pachelbel female; do
    for s in crickets cat-milk drums; do
        python khac.py $c $s --epochs 50 --lambd 500 --gamma 0 --pieces 4 --stack 0 --batch_size 8192
        python khac.py $c $s --epochs 50 --lambd 500 --gamma 0 --pieces 2 --stack 0 --channels 64
    done
done