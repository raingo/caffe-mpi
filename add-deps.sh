#!/bin/sh
# vim ft=sh



BIN=$1
dist_dir=$2
BIN_NAME=`basename $BIN`

scp $BIN cycle3:tmp
ssh cycle3 "ldd tmp/$BIN_NAME | grep not | awk '{print \$1}'" > $dist_dir/.deps

mkdir -p $dist_dir/deps

for dep in `cat $dist_dir/.deps`
do
    dep=`basename $dep`
    ldd $BIN | grep $dep | awk '{print $3}' | xargs -L 1 -I {} cp {} $dist_dir/deps
done
