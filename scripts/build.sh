#!/bin/bash
mkdir -p ../bin
mkdir -p ../build && cd "$_"
cmake ..
cmake --build .
mv ./src/mnist-mlp ../bin/mnist-mlp
