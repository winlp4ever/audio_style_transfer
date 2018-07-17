#! /usr/bin/env bash
for i in 0.1 0.2 0.3 0.4 0.0 0.5 0.6 0.7 0.8 0.9; do
	python khac.py 'gui_solo_mono' 'exo_flute' --lambd $i --gamma 2.0 --epochs 100 --cont_lyrs 20 --cmt 'ori'
done
