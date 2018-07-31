#! /usr/bin/env bash
btc=$((16384 * 2))
for s in cat-milk crickets drums male bongo-loop orchestra; do
    for l in 10 50 25 100; do
        for g in 1 0.2 0; do
            python acrossLayers.py $1 $s --lambd $l --gamma $g --epochs 100 --cont_lyrs 25 --stack 0 --batch_size $btc
        done
    done
done