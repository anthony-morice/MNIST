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
  bool test = false;
  if (argc == 5 && (strcmp(argv[3], "load-mlp") == 0 || strcmp(argv[3], "test-mlp") == 0)) {
    if (strcmp(argv[3], "test-mlp") == 0)
      test = true;
    std::cout << "Loading weights from file...";
    mlp = new MLP(argv[4]);
    std::cout << "DONE" << std::endl;
  } else {
    int input_dim = data.image_dims.first * data.image_dims.second;
    mlp = new MLP(input_dim, 300, data.num_classes);
  } // else
  while (true) {
    std::vector<std::pair<int,int>> prediction_results;
    std::cout << "Making predictions...";
    std::cout.flush();
    for (int i = 0; i < data.num_images; i++)
      prediction_results.push_back({mlp->predict(data.get_image(i)), data.get_label(i)});
    std::cout << "DONE" << std::endl;
    write_to_csv("../results/prediction_results.csv", prediction_results);
    std::system("./confusion.py ../results/prediction_results.csv");
    if (test)
      break;
    std::cout << "Continue Training? [y/n] ";
    char response;
    std::cin >> response;
    if (response != 'y')
      break;
    int epochs;
    float lr, dr;
    std::cout << "# Epochs? ";
    std::cin >> epochs;
    std::cout << "Learning rate? (0,1.0] ";
    std::cin >> lr;
    std::cout << "Decay rate? (0,1.0) ";
    std::cin >> dr;
    auto[losses, validation_accuracies] = mlp->fit(data, epochs, 32, lr, dr);
    mlp->save_weights("../results/mlp_weights.xml");
    write_to_csv("../results/training_losses.csv", losses);
    write_to_csv("../results/validation_accuracies.csv", validation_accuracies);
    std::system("./graph.py ../results/training_losses.csv training_losses");
    std::system("./graph.py ../results/validation_accuracies.csv validation_accuracies");
  } // while
  delete mlp; // cleanup
  return 0;
} // main()
