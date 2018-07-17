#! /usr/bin/env bash
for i in $( seq 1 10); do
	python khac.py 'bass' 'flute' --lambd $i --gamma $(( 1 + i % 3 )) --epochs 100 --cont_lyrs 20
done
