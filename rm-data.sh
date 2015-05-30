#!/bin/sh
# vim ft=sh

target=/localdisk/yli-data/caffe-ps
for node in `cat nodefile | sort | uniq`
do
    ssh $node "rm -Rf $target"
done
