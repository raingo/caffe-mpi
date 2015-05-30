#!/bin/sh
# vim ft=sh


log_dir='./logs-2015-05-29_14-12-52/logs/'
python plot2.py $log_dir/cifar10_quick
python plot2.py $log_dir/cifar10_full
python plot2.py $log_dir/lenet
