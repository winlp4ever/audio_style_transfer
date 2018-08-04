#!/usr/bin/env bash
for s in crickets drums gui_solo_mono exo_flute cat-milk; do
    for g in 10 0 20 40; do
        python acrossLayers.py $1 $s --epochs 100 --stack 0 --cont_lyrs 25 --batch_size $((16384*2)) --lambd 50 --gamma $g --cmt main_test
    done
done