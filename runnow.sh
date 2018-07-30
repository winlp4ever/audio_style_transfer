#! /usr/bin/env bash
btc=$((16384 * 2))
for c in pachelbel female; do
    for s in cat-milk crickets drums; do
        python khac.py $c $s --lambd 500 --gamma 100 --epochs 100 --cont_lyrs 25 --stack 0 --duration 2 --batch_size 8192
    done
done