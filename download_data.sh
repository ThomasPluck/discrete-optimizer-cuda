#!/bin/sh

while getopts d: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
    esac
done

if ["${dataset,,}" == "mnist"]
    then

    mkdir .data && cd .data
    mkdir mnist && cd mnist
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

    for file in *.gz; do tar xzvf "${file}" && rm "${file}"; done

fi