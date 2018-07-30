#! /usr/bin/env bash
for s in 0 1 2 -1; do
    for l in 100 50 10 5; do
        for g in 10 50 0 100; do
            for c in cat-milk crickets drums; do
                command="python khac.py pachelbel $c --epochs 100 --lambd $l --gamma $g --cont_lyrs 27 --batch_size $(( 16384 * 3 ))"
                if [ $s -ge 0 ]; then
                    $command --stack $s
                else
                    $command
                fi
            done
        done
    done
done