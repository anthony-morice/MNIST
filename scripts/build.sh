#!/bin/bash
mkdir -p ../bin
mkdir -p ../build && cd "$_"
cmake ..
cmake --build .
mv ./src/mnist-mlp ../bin/mnist-mlp
cp ../scripts/confusion.py ../bin/confusion.py
cp ../scripts/graph.py ../bin/graph.py
