#! /usr/bin/env bash
btc=$((16384 * 2))
for l in 100 200 500; do
    for g in 20 0 10 40 100; do
        for s in drums crickets cat-milk gui_solo_mono exo_flute; do
            python acrossLayers.py $1 $s --epochs 100 --stack 0 --cont_lyrs 25 --batch_size $((16384*2)) --lambd $l --gamma $g --cmt main_test
        done
    done
done