#include <iostream>
#include "mnist.h"
#include "mlp.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "USAGE: mlp-test <path-to-imgs> <path-to-labels>" << std::endl;
    return 1;
  } // if
  Mnist data(argv[1], argv[2]); // load and parse data
  MLP mlp(4, 3, 2);
  return 0;
} // main()
