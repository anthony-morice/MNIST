#include "mnist.h"
#include "mlp.h"
#include <fstream>
#include <iostream>

void write_to_csv(std::string filename, std::vector<float>& v) {
  std::ofstream out(filename);
  for (int i = 0; i < (int) v.size(); i++)
    out << i << ',' << v[i] << std::endl;
  out.close();
} // write_to_csv()

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "USAGE: mnist-mlp <path-to-imgs> <path-to-labels>" << std::endl;
    return 1;
  } // if
  Mnist data(argv[1], argv[2]); // load and parse data
  int input_dim = data.image_dims.first * data.image_dims.second;
  MLP mlp(input_dim, 64, data.num_classes);
  std::vector<float> losses = mlp.fit(data, 3000, 32, 0.7, 0.9);
  write_to_csv("training_losses.csv", losses);
  for (int i = 0; i < 100; i++) {
    std::cout << "Prediction: " << mlp.predict(data.get_image(i))[0] << std::endl;
    data.view(i);
  } // for

  return 0;
} // main()
