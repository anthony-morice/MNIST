#!/bin/bash
if [ $# -ne 0 ] && [ $1 = "test" ]; then
  export GTEST_COLOR=1
  cd ../build && ctest --verbose
elif [ $# -eq 2 ] && [ $1 = "load" ]; then
  ../bin/mnist-mlp ../dataset/t10k-images.idx3-ubyte ../dataset/t10k-labels.idx1-ubyte load $2
else
  ../bin/mnist-mlp ../dataset/train-images.idx3-ubyte ../dataset/train-labels.idx1-ubyte
fi
