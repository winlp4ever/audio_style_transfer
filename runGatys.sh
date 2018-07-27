#!/usr/bin/env bash
for i in 0 100 400; do
    for j in 1 5 10 50; do
        for s in cat-milk crickets drums; do
            for c in pachelbel female; do
                python gatys.py $c $s --epochs 50 --lambd $j --gamma $i --cont_lyrs 25 --style_lyrs 0 1 2 3 4 5 6 7 8 9 --batch_size 32768
            done
        done
    done
done