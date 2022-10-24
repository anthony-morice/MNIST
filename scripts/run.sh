#!/bin/bash
if [ $# -ne 0 ] && [ $1 = "test" ]; then
  export GTEST_COLOR=1
  cd ../build && ctest --verbose
else
  ../bin/mnist-mlp ../dataset/train-images.idx3-ubyte ../dataset/train-labels.idx1-ubyte
fi
