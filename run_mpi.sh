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
snap_file=logs/`basename $model`.$NumberOfProcesses.bin

mpirun -np $NumberOfProcesses ./bin/sgd-mpi  -lr 0.001 -snap_path $snap_file -model $model -iterations 10000 | tee $log_file
