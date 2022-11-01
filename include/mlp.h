/**
 * @file mlp.h
 */

#ifndef MLP_H_
#define MLP_H_

#include <mnist.h>
#include <vec2df.h>
#include <opencv2/core/mat.hpp>
#include <vector>
#include <utility>
#include <string>
#include <tuple>

class MLP {
  public:
    MLP(int n_input, int n_hidden, int n_output);
    MLP(std::string weight_file);
    std::pair<std::vector<float>, std::vector<float>> fit(Mnist& mnist, int n_epochs = 100, 
        int batch_size = 32, float lr = 0.7, float dr = 0.98);
    int predict(const cv::Mat& img);
    std::vector<int> predict_mnist(Mnist& mnist);
    void save_weights(std::string file) const;
    void load_weights(std::string file);
    static vec2df fc(const vec2df& x, vec2df& w, const vec2df& b);
    static std::tuple<vec2df, vec2df, vec2df> fc_backward(vec2df& dl_dy,
        const vec2df& x, const vec2df& w, const vec2df& b, const vec2df& y);
    static vec2df relu(const vec2df& x);
    static vec2df relu_backward(const vec2df& dl_dy, const vec2df& x, const vec2df& y);
    static std::pair<float, vec2df> loss_cross_entropy_softmax(const vec2df& x, const vec2df& y);
    int predict(const vec2df& x);

  private:
    vec2df w1;
    vec2df b1;
    vec2df w2;
    vec2df b2;
    void load_training_data(std::vector<std::vector<std::pair<vec2df,vec2df>>>& v, 
        Mnist& mnist, int batch_size);
};

#endif // MLP_H_
