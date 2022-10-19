#include "mlp.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
#include <tuple>
#include <vector>
#include <iostream>

MLP::MLP(int n_input, int n_hidden, int n_output) {
  // initialize weights and biases using gaussian(0,1) noise
  vec2df w1(n_input, n_hidden); 
  w1.gaussian_fill();
  this->w1 = w1;
  vec2df w2(n_hidden, n_output); 
  w2.gaussian_fill();
  this->w2 = w2;
  vec2df b1(1, n_hidden); 
  b1.gaussian_fill();
  this->b1 = b1;
  vec2df b2(1, n_output); 
  b2.gaussian_fill();
  this->b2 = b2;
} // MLP()

std::vector<int> MLP::predict(const cv::Mat& img) const {
  vec2df x(img.size[0] * img.size[1], (float*) img.data);
  return {fc(relu(fc(x, this->w1, this->b1)), this->w2, this->b2).argmax()};
} // predict()

std::vector<int> MLP::predict(Mnist& mnist) const {
  std::vector<int> predictions(mnist.num_images);
  for (int i = 0; i < mnist.num_images; i++)
    predictions[i] = predict(mnist.get_image(i))[0];
  return predictions;
} // predict()

void MLP::load_training_data(std::vector<std::vector<std::pair<vec2df,vec2df>>>& v, Mnist& mnist, int batch_size) {
  auto mini_batches = mnist.get_mini_batches(batch_size);
  for (int i = 0; i < (int) mini_batches.size(); i++) {
    v.push_back({});
    for (int t : mini_batches[i]) {
      const cv::Mat& img = mnist.get_image(t); 
      cv::Mat norm_img;
      cv::normalize(img, norm_img, 1.0, 0.0, cv::NORM_L1);
      vec2df x(norm_img.size[0] * norm_img.size[1], (float*) norm_img.data);
      vec2df y = mnist.get_onehot(t);
      v[i].push_back({x,y});
    } // for t
  } // for i 
} // load_training_data()

std::vector<float> MLP::fit(Mnist& mnist, int n_iter, int batch_size, float lr, float dr) {
  std::vector<std::vector<std::pair<vec2df, vec2df>>> mini_batches;
  this->load_training_data(mini_batches, mnist, batch_size);
  int batch_total = mini_batches.size();
  std::vector<float> losses;
  int batch = 0;
  std::cout << "\n*******Training Multi-layer Perceptron*******\n" << std::endl;
  // allocate space for gradient accumulators
  vec2df dL_dw1(this->w1.get_shape().second, this->w1.get_shape().first);
  vec2df dL_db1(this->b1.get_shape());
  vec2df dL_dw2(this->w2.get_shape().second, this->w2.get_shape().first);
  vec2df dL_db2(this->b2.get_shape());
  // mini-batched stochastic gradient descent
  for (int i = 0; i < n_iter; i++) {
    if (i != 0 && i % 500 == 0) { // update learning rate
      lr *= dr;
      std::cout << "\nIteration: " << i << " of " << n_iter << std::endl;
      std::cout << "  new lr: " << lr << ", loss: " << *losses.rbegin() << std::endl;
    } // if
    // zero gradient accumulators
    dL_dw1.zeros_fill();
    dL_db1.zeros_fill();
    dL_dw2.zeros_fill();
    dL_db2.zeros_fill();
    float loss_sum = 0;
    auto& mini_batch = mini_batches[batch];
    for (auto& [x, y] : mini_batch) {
      // forward_pass
      vec2df a1 = fc(x, this->w1, this->b1);
      vec2df f1 = relu(a1);
      vec2df y_tilde = fc(f1, this->w2, this->b2); // prediction
      auto [loss, dl_dy] = loss_cross_entropy_softmax(y_tilde, y);
      loss_sum += loss;
      // backpropagation
      auto [dl_df1, dl_dw2, dl_db2] = fc_backward(dl_dy, f1, w2, b2, y);
      vec2df dl_da1 = relu_backward(dl_df1, a1, f1);
      auto [dl_dx, dl_dw1, dl_db1] = fc_backward(dl_da1, x, w1, b1, a1);
      // gradient accumulation
      dL_dw1 += dl_dw1;
      dL_db1 += dl_db1;
      dL_dw2 += dl_dw2;
      dL_db2 += dl_db2;
    } // for 
    // increment batch index or reset if at end of epoch
    batch += batch + 2 < batch_total ? 1 : 0;
    // update weights
    this->w1 -= dL_dw1.scale(lr / mini_batch.size()).transpose();
    this->b1 -= dL_db1.scale(lr / mini_batch.size());
    this->w2 -= dL_dw2.scale(lr / mini_batch.size()).transpose();
    this->b2 -= dL_db2.scale(lr / mini_batch.size());
    losses.push_back(loss_sum / mini_batch.size());
  } // for i
  return losses;
} // fit()

vec2df MLP::fc(const vec2df& x, const vec2df& w, const vec2df& b) {
  return x * w + b;
} // fc()

std::tuple<vec2df, vec2df, vec2df> MLP::fc_backward(const vec2df& dl_dy, const vec2df& x,
    const vec2df& w, const vec2df& b, const vec2df& y) {
  vec2df dl_dx = dl_dy * w.transpose();
  vec2df dl_dw = dl_dy.transpose() * x;
  vec2df dl_db = dl_dy;
  return std::make_tuple(dl_dx, dl_dw, dl_db);
} // fc_backward()

vec2df MLP::relu(const vec2df& x) {
  return vec2df::clip_min(x, 0);  
} // relu()

vec2df MLP::relu_backward(const vec2df& dl_dy, const vec2df& x, const vec2df& y) {
  return vec2df::element_multiply(dl_dy, vec2df::clip_max(y, 1)); 
} // relu_backward()

std::pair<float, vec2df> MLP::loss_cross_entropy_softmax(const vec2df& x, const vec2df& y) {
  vec2df soft = vec2df::softmax(x);
  float loss = -1 * log(soft.get(y.argmax()));
  vec2df dl_dy = soft - y;
  return {loss, dl_dy}; 
} // loss_cross_entropy_softmax()
