#include <iostream>
#include "mnist.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cout << "USAGE: mnist-read-test <train> <train-lab> <test> <test-lab>" << std::endl;
    return(1);
  } // if
  Mnist data(argv[1], argv[2], argv[3], argv[4]);
  return 0;
} // main()
