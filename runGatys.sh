#!/usr/bin/env bash
for c in pachelbel female canon; do
    for s in gui_solo_mono crickets drums cat-milk exo_flute bongo-loop; do
        for g in 10 20 50; do
            python gatys.py $c $s --epochs 100 --cont_lyrs 25 --stack 0 --batch_size $((16384*2)) --lambd 100 --gamma $g
        done
    done
done