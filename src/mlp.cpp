#include "mlp.h"
#include <iostream>
#include <tuple>

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

static vec2df MLP::fc(const vec2df& x, const vec2df& w, const vec2df& b) {
  return x * w + b;
} // fc()

static std::tuple<vec2df, vec2df, vec2df> MLP::fc_backward(const vec2df& dl_dy, const vec2df& x,
    const vec2df& w, const vec2df& b, const vec2df& y) {
  // determine dl_dx
  vec2df dl_dx = w * dl_dy;
  // determine dl_dw
  //TODO
  // determine dl_db
  vec2df dl_db = dl_dy;
  return std::make_tuple(dl_dx, dl_dw, dl_db);
}
