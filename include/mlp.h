/**
 * @file mlp.h
 */

#ifndef MLP_H_
#define MLP_H_

#include "mnist.h"
#include <opencv2/core/mat.hpp>
#include <vector>
#include <utility>

class MLP {
  public:
    MLP(int n_input, int n_hidden, int n_output);
    static void print_weight(const std::vector<std::vector<float>>& w);
    static void print_bias(const std::vector<float>& b);
    /*
    void fit(std::vector<cv::Mat> imgs, int n_iter, int batch_size = 32, int lr = 0.1, int dr = 0.9);
    std::vector<int> predict(std::vector<cv::Mat> imgs);
    static std::vector<float> fc(const std::vector<float>& x, const std::vector<std::vector<float>>& w,
                                 const std::vector<float>& b);
    static std::vector<float> fc_backward(const std::vector<float>& dl_dy, const std::vector<float>& x,
                                          const std::vector<std::vector<float>>& w, const std::vector<float>& b,
                                          const std::vector<float>& y);
    static std::vector<float> relu(const std::vector<float>& x);
    static std::vector<float> relu_backward(const std::vector<float>& dl_dy, const std::vector<float>& x, 
                                            const std::vector<float>& y);
    static std::pair<float, std::vector<float>> loss_cross_entropy_softmax(const std::vector<float>& x,
                                                                           const std::vector<float>& y);
    */

  private:
    std::vector<std::vector<float>> w1;
    std::vector<float> b1;
    std::vector<std::vector<float>> w2;
    std::vector<float> b2;
};

#endif // MLP_H_
