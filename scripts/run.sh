#!/bin/bash
cd ../bin
if [ $# -ne 0 ] && [ $1 = "test" ]; then
  export GTEST_COLOR=1
  cd ../build && ctest --verbose
elif [ $# -eq 2 ] && [ $1 = "load-mlp" ]; then
  ./mnist-mlp ../dataset/train-images.idx3-ubyte ../dataset/train-labels.idx1-ubyte $1 $2
elif [ $# -eq 2 ] && [ $1 = "test-mlp" ]; then
  ./mnist-mlp ../dataset/t10k-images.idx3-ubyte ../dataset/t10k-labels.idx1-ubyte $1 $2
else
  ./mnist-mlp ../dataset/train-images.idx3-ubyte ../dataset/train-labels.idx1-ubyte
fi
