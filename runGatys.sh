#!/usr/bin/env bash
for s in gui_solo_mono crickets drums cat-milk exo_flute bongo-loop; do
    for g in 10 20 50; do
            python gatys.py $1 $s --epochs 100 --cont_lyrs 25 --batch_size $((16384*2)) --lambd 100 --gamma $g
    done
done
