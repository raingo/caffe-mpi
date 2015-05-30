#!/bin/sh
# vim ft=sh

target=/localdisk/yli-data/caffe-ps
for node in `cat nodefile | sort | uniq`
do
    ssh $node "mkdir -p $target"
    rsync -carL dist/ $node:$target
done
