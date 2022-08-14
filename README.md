# Build instructions
```
mkdir build
cd build
cmake ..
cmake --build .
./test-mnist-read ../dataset/train-images.idx3-ubyte ../dataset/train-labels.idx1-ubyte ../dataset/t10k-images.idx3-ubyte ../dataset/t10k-labels.idx1-ubyte
```

# Recompile after changes
```
cd build
cmake --build .
```
