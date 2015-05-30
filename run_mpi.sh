#!/bin/sh
# vim ft=sh

NumberOfProcesses=$1
model=$2
eval=$3

if [ -z "$model" ]
then
    echo $0 np model
    exit
fi

mkdir -p logs
log_file=logs/`basename $model`.$NumberOfProcesses.log

mpirun -np $NumberOfProcesses ./bin/sgd-mpi -eval_iter $eval -lr 0.001 -model $model -iterations 10000 | tee $log_file
