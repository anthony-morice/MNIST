#include "mlp.h"
#include <random>
#include <iostream>

void MLP::print_weight(const std::vector<std::vector<float>>& w) {
  std::cout << "[" << std::endl;
  for (auto row : w) {
    std::cout << "  [ ";
    for (auto val : row)
      std::cout << val << " ";
    std::cout << ']' << std::endl;
  } // for row
  std::cout << ']' << std::endl << std::endl;
} // print_weight()

void MLP::print_bias(const std::vector<float>& b) {
  std::cout << "[ ";
  for (auto val: b)
    std::cout << val << " ";
  std::cout << ']' << std::endl << std::endl;
} // print_weight()

MLP::MLP(int n_input, int n_hidden, int n_output) {
  // initialize weights and biases using gaussian(0,1) noise
  std::default_random_engine generator;
  std::normal_distribution<float> gaussian(0.0, 1.0);
  this->w1 = std::vector<std::vector<float>>(n_hidden, std::vector<float>(n_input));
  this->w2 = std::vector<std::vector<float>>(n_output, std::vector<float>(n_hidden));
  this->b1 = std::vector<float>(n_hidden);
  this->b2 = std::vector<float>(n_output);
  bool b2_flag = false;
  for (int i = 0; i < n_hidden; i++) {
    this->b1.at(i) = gaussian(generator);
    for (int j = 0; j < n_input; j++)
      this->w1.at(i).at(j) = gaussian(generator);
    for (int j = 0; j < n_output; j++) {
      this->w2.at(j).at(i) = gaussian(generator);
      if (!b2_flag)
        this->b2.at(j) = gaussian(generator);
    } // for j
    b2_flag = true;
  } // for
  std::cout << "w1: " << std::endl;
  MLP::print_weight(this->w1);
  std::cout << "w2: " << std::endl;
  MLP::print_weight(this->w2);
  std::cout << "b1: " << std::endl;
  MLP::print_bias(this->b1);
  std::cout << "b2: " << std::endl;
  MLP::print_bias(this->b2);
} // MLP()

/*
void fit(vector<cv::Mat> imgs, int n_iter, int batch_size, int lr, int dr) {
  // TODO
} // fit()

vector<int> predict(vector<cv::Mat> imgs) {
  // TODO
} // predict()

std::vector<float> fc(const vector<float>& x, const vector<float>& w, const vector<float>& b) {
  // TODO
} // fc()

std::vector<float> fc_backward(const vector<float>& dl_dy, const vector<float>& x,
                                      const vector<float>& w, const vector<float>& b,
                                      const vector<float>& y) {
  // TODO
} // fc_backward()

std::vector<float> relu(const vector<float>& x) {
  // TODO
} // relu()

std::vector<float> relu_backward(const vector<float>& dl_dy, const vector<float>& x, 
                                        const vector<float>& y) {
  // TODO
} // relu_backward()

std::pair<float, std::vector<float>> loss_cross_entropy_softmax(x, y) {
  // TODO
} // loss_cross_entropy_softmax()
*/
