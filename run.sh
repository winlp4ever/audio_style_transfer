#!/usr/bin/env bash
for i in 0 2 4 6
do
    python NetFeat.py "flute1$i" 2 0 --length 4096 -e 50 --k 5
done