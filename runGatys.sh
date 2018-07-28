#!/usr/bin/env bash
for s in cat-milk crickets drums; do
    for l in 0.5 1 5 10; do
        for g in 1 10 20 50 100; do
            python gatys.py female $s --lambd $l --gamma $g --epochs 100 --batch_size $(( 16384 * 3 ))
        done
    done
done