This repository contains an MNIST image/label parser and
multi-layer perceptron (MLP) classifier written from scratch in C++. 

OpenCV is used to view and normalize images; the googletest framework is used
for unit testing; OpenMP is used to parallelize MLP training; CBLAS is used for
matrix multiplications; and CMake is used as the build system.

MNIST data files can be found at
[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

![MLP Confusion Matrix](results/models/model_10-26-1733/model_10-26-1733_confusion.png)

# Usage instructions
## Build

```
cd scripts
./build.sh
```
## Run

To train a model from scratch...
```
cd scripts
./run.sh
```

To train a model from a checkpoint...
```
cd scripts
./run.sh load-mlp <path-to-checkpoint>
```

Or to load an existing model and evaluate on test images...
```
cd scripts
./run.sh test-mlp <path-to-checkpoint>
```

## Run Google Tests
```
cd scripts
./run.sh test
```
