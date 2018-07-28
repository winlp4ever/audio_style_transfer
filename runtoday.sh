#!/usr/bin/env bash
for s in cat-mil crickets drums; do
    for l in 0.5 1 5 10; do
        for g in 1 10 20 50 100; do
            python khac.py pachelbel $s --lambd $l --gamma $g --epochs 100
        done
    done
done