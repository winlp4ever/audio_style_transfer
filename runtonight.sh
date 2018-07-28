#! /usr/bin/env bash
for s in $( seq -1 2 ); do
    for l in 100 50 10 0; do
        for c in cat-milk crickets drums; do
            command="python khac.py pachelbel $c --epochs 100 --lambd $l --gamma 10 --cont_lyrs 27 --batch_size $(( 16384 * 3 ))"
            if [ $s -lt 0 ]; then
                $command
            else
                $command --stack $s
            fi
        done
    done
done