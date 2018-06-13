#!/usr/bin/env bash
for i in 1 3 5
do
    python NetFeat.py "flute1$i" 2 0 --length 4096 -e 50 --k 5
done