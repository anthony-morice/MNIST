#include <mnist.h>
#include <mlp.h>
#include <fstream>
#include <iostream>
#include <cstdlib>

void write_to_csv(std::string filename, std::vector<float>& v) {
  std::ofstream out(filename);
  for (float& i : v)
    out << i << std::endl;
  out.close();
} // write_to_csv()

void write_to_csv(std::string filename, std::vector<std::pair<int,int>>& v) {
  std::ofstream out(filename);
  for (int i = 0; i < (int) v.size(); i++)
    out << v[i].first << ',' << v[i].second << std::endl;
  out.close();
} // write_to_csv()

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "USAGE: mnist-mlp <path-to-imgs> <path-to-labels>" << std::endl;
    return 1;
  } // if
  Mnist data(argv[1], argv[2]); // load and parse data
  int input_dim = data.image_dims.first * data.image_dims.second;
  MLP mlp(input_dim, 128, data.num_classes);
  std::vector<float> losses = mlp.fit(data, 200000, 32, 0.75, 0.92);
  write_to_csv("../results/training_losses.csv", losses);
  std::system("./graph_losses.py ../results/training_losses.csv");
  std::vector<std::pair<int,int>> prediction_results;
  for (int i = 0; i < data.num_images; i++)
    prediction_results.push_back({mlp.predict(data.get_image(i)), data.get_label(i)});
  write_to_csv("../results/prediction_results.csv", prediction_results);
  std::system("./confusion.py ../results/prediction_results.csv");
  return 0;
} // main()
