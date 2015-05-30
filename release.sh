#!/bin/sh
# vim ft=sh

dist_dir=dist

apps=(sgd sgd-mpi)

rm -Rf $dist_dir

mkdir -p $dist_dir/bin


for app in ${apps[@]}
do
    echo $app
    cp $app $dist_dir/bin
    ./add-deps.sh $dist_dir/bin/$app $dist_dir
done

# deps for so file
for lib in `find $dist_dir/deps/ -name '*.so'`
do
    echo $lib
    ./add-deps.sh $lib $dist_dir
done

./extra-deps.sh $dist_dir
