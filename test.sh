#!/usr/bin/env bash
layers=($5 $6 $7 $8 $9)
[[ ${#layers[@]} -ne 0 ]] && params+="--layers ${layers[*]}"
if [ $4 = true ]
    then params+=" --output_file"
elif [ $4 = false ]
    then params+=
else
    echo "error" >&2
    exit 1
fi  

for i in 1 2 3
do 
    [[ -n ${!i} ]] && python compare.py "${!i}" --nb_exs 200 $params
done
