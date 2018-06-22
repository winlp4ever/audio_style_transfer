#!/usr/bin/env bash
for i in 1 2
do
   [[ -n ${!i} ]] && echo 'stupid'
done

