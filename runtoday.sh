#!/usr/bin/env bash
for s in cat-milk crickets drums; do
    for l in 10 5 20 2 1; do
        for g in 10 1 20 50 100; do
            python khac.py female $s --lambd $l --gamma $g --epochs 100 --batch_size $((16384 * 3)) --stack 0
        done
    done
done