MNIST training and test image/label parsing for machine learning projects.

## Build instructions
```
mkdir build
cd build
cmake ..
cmake --build .
./test-mnist-read <path-to-MNIST-image-file> <path-to-MNIST-label-file>
```

## Recompile after changes
```
cd build
cmake --build .
```
