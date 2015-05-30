#!/bin/sh
# vim ft=sh

dist_dir=`pwd`
export LD_LIBRARY_PATH=$dist_dir/deps:$LD_LIBRARY_PATH
#export PATH=$dist_dir/deps:$PATH
#export DYLD_LIBRARY_PATH=$dist_dir/deps:$DYLD_LIBRARY_PATH

ts=`date +'%Y-%m-%d_%H-%M-%S'`

for model in `find example -name '*.prototxt'`
do
    for np in `seq 8 -1 2`
    do
        ./run_mpi.sh $np $model
    done
    ./run_exp.sh $model
done

rsync logs gpu-cs:/home/yli/workspace/cs458/caffe-ps-v2/logs-$ts -ar
echo '$0' | mail -s "done" yli
