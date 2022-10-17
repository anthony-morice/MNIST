#include "vec2df.h"
#include <gsl/gsl_cblas.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <random>
#include <chrono>

vec2df::vec2df(int nrow, int ncol) {
  this->shape= {nrow, ncol};
  this->size = nrow * ncol;
  this->data = new float[this->size];
} // vec2df(int, int)

vec2df::vec2df(std::pair<int, int> shape) : vec2df(shape.first, shape.second) {}

vec2df::vec2df(const vec2df& other) {
  this->shape = other.shape;
  this->size = other.size;
  this->data = new float[this->size];
  for (int i = 0; i < this->size; i++)
    this->data[i] = other.data[i];
} // vec2df() copy constructor

vec2df& vec2df::operator=(const vec2df& other) {
  this->shape = other.shape;
  this->size = other.size;
  this->copy_data(other.data);
  return *this;
} // copy assignment operator

void vec2df::copy_data(float* data) {
  delete[] this->data;
  this->data = new float[this->size];
  for (int i = 0; i < this->size; i++)
    this->data[i] = data[i];
} // copy_data()

vec2df::~vec2df() {
  delete[] this->data;
} // ~vec2df()

const std::pair<int, int>& vec2df::get_shape() const {
  return this->shape;
} // get_shape()

float vec2df::get(int i, int j) const {
  assert(i < this->shape.first && j < this->shape.second);
  int index = i * this->shape.second + j;
  return this->data[index];
} // get()

void vec2df::print() const {
  std::cout << "Shape: (" << this->shape.first << ", ";
  std::cout << this->shape.second << ")" << std::endl;
  std::cout << "[" << std::endl;
  for (int i = 0; i < this->shape.first; i++) {
    std::cout << "  [ ";
    for (int j = 0; j < this->shape.second; j++)
      std::cout << this->get(i, j) << " ";
    std::cout << ']' << std::endl;
  } // for row
  std::cout << ']' << std::endl << std::endl;
} // print()

void vec2df::gaussian_fill(float mean, float std) {
  unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> gaussian(mean, std);
  for (int i = 0; i < this->size; i++)
    this->data[i] = gaussian(generator);
} // gaussian_fill()

vec2df vec2df::operator+(const vec2df& rhs) {
  assert(this->shape == rhs.shape);
  vec2df res(this->shape);
  for (int i = 0; i < res.size; i++)
    res.data[i] = this->data[i] + rhs.data[i];
  return res;
} // operator+()

vec2df vec2df::operator-(const vec2df& rhs) {
  assert(this->shape == rhs.shape);
  vec2df res(this->shape);
  for (int i = 0; i < res.size; i++)
    res.data[i] = this->data[i] - rhs.data[i];
  return res;
} // operator-()

bool vec2df::operator==(const vec2df& rhs) {
  float epsilon = 0.0001;
  if (this->shape != rhs.shape) 
    return false;
  for (int i = 0; i < this->size; i++) {
    if (std::fabs(this->data[i] - rhs.data[i]) >= epsilon) 
      return false;
  } // for
  return true;
} // operator==()

vec2df vec2df::operator*(const vec2df& rhs) {
  assert(this->shape.second == rhs.shape.first);
  int columns_a = this->shape.second;
  int columns_b = rhs.shape.second;
  int rows_c = this->shape.first, columns_c = columns_b;
  vec2df res(rows_c, columns_c);
  float* a = this->data;
  float* b = rhs.data;
  float* c = res.data;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              rows_c, columns_c, columns_a, 1.0,
              a, columns_a, b, columns_b, 0.0, c, columns_c);
  return res;
} // operator*()
