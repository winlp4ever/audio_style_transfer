#!/usr/bin/env bash
for s in pachelbel female; do
    for c in crickets drums cat-milk bongo-loop exo_flute; do
        python acrossLayers.py $c $s --epochs 100 --cont_lyrs 24 --stack 0 --lambd 100 --gamma 20 --batch_size $((16384*2)) --channels 64
    done
done