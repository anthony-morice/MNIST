/**
 * @file mlp.h
 */

#ifndef MLP_H_
#define MLP_H_

#include "mnist.h"
#include "vec2df.h"
#include <opencv2/core/mat.hpp>
#include <vector>
#include <utility>
#include <string>
#include <tuple>

class MLP {
  public:
    MLP(int n_input, int n_hidden, int n_output);
    std::vector<float> fit(Mnist mnist, int n_iter, int batch_size = 32, int lr = 0.1, int dr = 0.9);
    std::vector<int> predict(cv::Mat img) const;
    std::vector<int> predict(Mnist mnist) const;

  private:
    vec2df w1;
    vec2df b1;
    vec2df w2;
    vec2df b2;
    static vec2df fc(const vec2df& x, const vec2df& w, const vec2df& b);
    static std::tuple<vec2df, vec2df, vec2df> fc_backward(const vec2df& dl_dy,
        const vec2df& x, const vec2df& w, const vec2df& b, const vec2df& y);
    static vec2df relu(const vec2df& x);
    static vec2df relu_backward(const vec2df& dl_dy, const vec2df& x, const vec2df& y);
    static std::pair<float, vec2df> loss_cross_entropy_softmax(const vec2df& x, const vec2df& y);
};

#endif // MLP_H_
