#! /usr/bin/env bash
for i in $( seq 1 10 ); do
	python show.py $1 $2 --channel $i
done
