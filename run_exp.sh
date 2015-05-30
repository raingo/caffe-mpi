#!/bin/sh
# vim ft=sh

model=$1

if [ -z "$model" ]
then
    echo $0 model
    exit
fi

mkdir -p logs
snap_file=logs/`basename $model`.bin

./bin/sgd -snap_path $snap_file -model $model -lr 0.001 -iterations 10000
