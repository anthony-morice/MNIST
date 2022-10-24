#include <mnist.h>
#include <mlp.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string.h>

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
    std::cout << "USAGE: mnist-mlp <path-to-imgs> <path-to-labels> [load] [path-to-weights]" << std::endl;
    return 1;
  } // if
  Mnist data(argv[1], argv[2]); // load and parse data
  MLP* mlp;
  if (argc == 5 && strcmp(argv[3], "load") == 0) {
    std::cout << "Loading weights from file...";
    mlp = new MLP(argv[4]);
    std::cout << "DONE" << std::endl;
  } else {
    int input_dim = data.image_dims.first * data.image_dims.second;
    mlp = new MLP(input_dim, 256, data.num_classes);
    std::vector<float> losses = mlp->fit(data, 10000, 32, 0.65, 0.95);
    mlp->save_weights("../results/weights_256-hidden.xml");
    write_to_csv("../results/training_losses.csv", losses);
    std::system("./graph_losses.py ../results/training_losses.csv");
  } // else
  std::vector<std::pair<int,int>> prediction_results;
  std::cout << "Making predictions...";
  for (int i = 0; i < data.num_images; i++)
    prediction_results.push_back({mlp->predict(data.get_image(i)), data.get_label(i)});
  std::cout << "DONE" << std::endl;
  write_to_csv("../results/prediction_results.csv", prediction_results);
  std::system("./confusion.py ../results/prediction_results.csv");
  delete mlp; // cleanup
  return 0;
} // main()
