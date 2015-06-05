#!/bin/sh
# vim ft=sh

#log_dir='./logs-2015-05-30_19-34-51/'
#log_dir='./logs-2015-05-31_12-18-23/'
log_dir='./logs-2015-06-02_11-38-11/'
#gpus="0 1"
gpus="0"
ngpus=`echo $gpus | wc -w`

find $log_dir -name '*.bin' | parallel -j$ngpus --xapply ./evaluator.sh :::: - ::: $gpus

echo $0 | mail -S 'done' yli
