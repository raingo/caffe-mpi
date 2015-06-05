#!/bin/sh
# vim ft=sh

snap_path=$1
gpu=$2
log_file=$snap_path.log

snap=`basename $snap_path | awk -F'.' '{print $1}'`
eval_iter=`cat example/configs | grep $snap | awk '{print $2}'`

./evaluator -gpu $gpu -snap_path $snap_path -model example/$snap.prototxt -eval_iter $eval_iter > $log_file
