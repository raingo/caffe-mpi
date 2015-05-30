#!/bin/sh
# vim ft=sh

model=$1

if [ -z "$model" ]
then
    echo $0 model
    exit
fi

mkdir -p logs
log_file=logs/`basename $model`.log

./bin/sgd -model $model -lr 0.001 -iterations 1000 | tee $log_file
