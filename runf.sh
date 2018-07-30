#!/usr/bin/env bash
for s in drums crickets voc_mono orchestra; do
    for c in pachelbel female; do
        python khac.py $c $s --epochs 100 --cont_lyrs 24 --stack 0 --lambd 500 --gamma 100 --batch_size 8192 --duration 2
    done
done