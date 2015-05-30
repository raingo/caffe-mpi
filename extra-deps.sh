#!/bin/sh
# vim ft=sh

dist_dir=$1
cp run*.sh $dist_dir/
cp mpi_env.csh $dist_dir/
ln -s `pwd`/example $dist_dir/
