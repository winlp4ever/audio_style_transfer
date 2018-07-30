#!/usr/bin/env bash
for p in 1 2 3; do
    for s in cat-milk crickets drums; do
        for l in 5 2 1; do
            for g in 1 0.5 0.2 0.05 0.01; do
                for c in pachelbel female; do
                    python acrossLayers.py female $s --lambd $l --gamma $g --epochs 100 --batch_size $((16384*$p)) --stack 0
                done
            done
        done
    done
done