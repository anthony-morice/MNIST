#include <iostream>
#include "mnist.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "USAGE: mnist-read-test <path-to-imgs> <path-to-labels>" << std::endl;
    return(1);
  } // if
  Mnist data(argv[1], argv[2]);
  data.view();
  return 0;
} // main()
