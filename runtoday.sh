#!/usr/bin/env bash
for i in 500 800 1000; do
    for j in 100 500; do
        for s in crickets cat-milk drums; do
            python khac.py 'pachelbel' $s --epochs 100 --cont_lyrs 27 --stack 0 --lambd $i --gamma $j --batch_size 8192 --pieces 4
            python khac.py 'pachelbel' $s --epochs 100 --cont_lyrs 27 --stack 0 --lambd $i --gamma $j --channels 64 --pieces 4
        done
    done
done