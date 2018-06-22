#!/usr/bin/env bash
for dir in ./data/aac/*/
do
    echo "working on dir $dir ..."
    for f in $dir*/*m4a
    do
        avconv -i "$f" "${f/%m4a/wav}"
    done
    echo "Done."
done
