#! /usr/bin/env bash
for i in $( seq 1 10 ); do
	for j in $( seq 1 10 ); do
		python khac.py 'gui_solo_mono' 'exo_flute' --lambd $i --gamma $j --epochs 100 --cont_lyrs $((20 + $i % 3))
	done
done
