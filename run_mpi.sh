#!/bin/sh
# vim ft=sh

NumberOfProcesses=$1
model=$2

if [ -z "$model" ]
then
    echo $0 np model
    exit
fi

mkdir -p logs
log_file=logs/`basename $model`.$NumberOfProcesses.log

mpirun -np $NumberOfProcesses ./bin/sgd-mpi -lr 0.001 -model $model -iterations 1000 | tee $log_file
