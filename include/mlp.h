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
    std::vector<float> fit(Mnist& mnist, int n_iter, int batch_size = 32, float lr = 0.1, float dr = 0.9);
    std::vector<int> predict(const cv::Mat& img) const;
    std::vector<int> predict(Mnist& mnist) const;

  private:
    vec2df w1;
    vec2df b1;
    vec2df w2;
    vec2df b2;
    vec2df feature_means;
    vec2df feature_stds;
    vec2df& normalize(vec2df& x);
    void normalize_all(std::vector<std::vector<std::pair<vec2df,vec2df>>>& x);
    void normalization_fit(std::vector<std::vector<std::pair<vec2df,vec2df>>>& x);
    void load_training_data(std::vector<std::vector<std::pair<vec2df,vec2df>>>& v, Mnist& mnist, int batch_size);
    static vec2df fc(const vec2df& x, const vec2df& w, const vec2df& b);
    static std::tuple<vec2df, vec2df, vec2df> fc_backward(const vec2df& dl_dy,
        const vec2df& x, const vec2df& w, const vec2df& b, const vec2df& y);
    static vec2df relu(const vec2df& x);
    static vec2df relu_backward(const vec2df& dl_dy, const vec2df& x, const vec2df& y);
    static std::pair<float, vec2df> loss_cross_entropy_softmax(const vec2df& x, const vec2df& y);
};

#endif // MLP_H_
