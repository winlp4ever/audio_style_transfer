#!/usr/bin/env bash
for s in crickets drums gui_solo_mono exo_flute cat-milk; do
    for l in 50 100 200; do
        for g in $(seq 0 50 10); do
            cmd="python gatys.py $1 $s --epochs 100 --cont_lyrs 25 --batch_size $((16384*2)) --lambd $l --gamma $g"
            $cmd
            $cmd --style_lyrs 17
        done
    done
done
