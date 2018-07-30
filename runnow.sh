#! /usr/bin/env bash
btc=$((16384 * 3))
for l in 10 50 5; do
    for g in 1 10 20 50 100; do
        for stk in $(seq 0 2); do
            for s in cat-milk orchestra crickets bongo-loop drums voc_mono; do
                python khac.py pachelbel $s --lambd $l --gamma $g --epochs 125 --stack 0 --batch_size $btc --cont_lyrs 27
            done
        done
    done
done