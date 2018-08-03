#! /usr/bin/env bash
for c in pachelbel canon female; do
    for s in gui_solo_mono crickets drums cat-milk exo_flute; do
        for l in 100 200 500; do
            for g in 40 60 100; do
                python acrossLayers.py $c $s --epochs 100 --cont_lyrs 25 --stack 0 --batch_size $((16384*2)) --lambd $l --gamma $g --cmt main_test
            done
        done
    done
done