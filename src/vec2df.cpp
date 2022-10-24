#include "vec2df.h"
#include <gsl/gsl_cblas.h>
#include <iostream>
#include <cassert>
#include <random>
#include <chrono>
#include <cstring>
#include <cmath>

vec2df::vec2df(int nrow, int ncol) {
  this->shape = {nrow, ncol};
  this->size = nrow * ncol;
  this->data = new float[this->size];
} // vec2df(int, int)

vec2df::vec2df(std::pair<int, int> shape) : vec2df(shape.first, shape.second) {}

vec2df::vec2df(int size, float* data) { 
  this->shape = {1, size};
  this->size = size;
  this->data = new float[this->size];
  for (int i = 0; i < this->size; i++)
    this->data[i] = data[i];
} // vec2df(int, float*)

vec2df::vec2df(std::vector<int> v) {
  this->shape = {1, v.size()};
  this->size = v.size();
  this->data = new float[this->size];
  for (int i = 0; i < this->size; i++)
    this->data[i] = (float) v[i];
} // vec2df()

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
  delete[] this->data;
  this->data = new float[this->size];
  for (int i = 0; i < this->size; i++)
    this->data[i] = other.data[i];
  return *this;
} // copy assignment operator

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
} // get(int, int)

float vec2df::get(int i) const {
  return this->data[i];
} // get(int)

float& vec2df::at(int i, int j) const {
  assert(i < this->shape.first && j < this->shape.second);
  int index = i * this->shape.second + j;
  return this->data[index];
} // at(int, int)

float& vec2df::at(int i) const {
  return this->data[i];
} // at(int)

int vec2df::argmax() const {
  int a_max = 0;
  float val_max = this->data[0];
  for (int i = 1; i < this->size; i++) {
    if (this->data[i] > val_max) {
      val_max = this->data[i];
      a_max = i;
    }// if
  } // for
  return a_max;
} // argmax()

int vec2df::argmin() const {
  int a_min = 0;
  float val_min = this->data[0];
  for (int i = 1; i < this->size; i++) {
    if (this->data[i] < val_min) {
      val_min= this->data[i];
      a_min = i;
    }// if
  } // for
  return a_min;
} // argmax()

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

vec2df vec2df::transpose() const {
  vec2df res(this->shape.second, this->shape.first);
  for (int i = 0; i < this->shape.first; i++) {
    for (int j = 0; j < this->shape.second; j++) {
      res.at(j, i) = this->get(i,j);
    } // for j
  } // for i
  return res;
} // transpose()

void vec2df::gaussian_fill(float mean, float std) {
  unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> gaussian(mean, std);
  for (int i = 0; i < this->size; i++)
    this->data[i] = gaussian(generator);
} // gaussian_fill()

void vec2df::zeros_fill() {
  std::memset(this->data, 0, sizeof(float) * this->size);
} //zeros_fill()

vec2df vec2df::clip_min(const vec2df& a, float min_val) {
  vec2df res(a.shape);
  for (int i = 0; i < res.size; i++) {
    if (a.data[i] < min_val)
      res.data[i] = min_val;
    else
      res.data[i] = a.data[i];
  } // for
  return res;
} // clip_min()

vec2df vec2df::clip_max(const vec2df& a, float max_val) {
  vec2df res(a.shape);
  for (int i = 0; i < res.size; i++) {
    if (a.data[i] > max_val)
      res.data[i] = max_val;
    else
      res.data[i] = a.data[i];
  } // for
  return res;
} // clip_max()

vec2df& vec2df::scale(float f) {
  for (int i = 0; i < this->size; i++)
    this->data[i] *= f;
  return *this;
} // scale()

vec2df& vec2df::sqrt() {
  for (int i = 0; i < this->size; i++)
    this->data[i] = std::sqrt(this->data[i]);
  return *this;
} // sqrt()

vec2df& vec2df::add(float f) {
  for (int i = 0; i < this->size; i++)
    this->data[i] += f;
  return *this;
} // add()

vec2df vec2df::element_multiply(const vec2df& a, const vec2df& b) {
  assert(a.shape == b.shape);
  vec2df res(a.shape);
  for (int i = 0; i < res.size; i++)
    res.data[i] = a.data[i] * b.data[i];
  return res;
} // element_multiply()


vec2df vec2df::softmax(const vec2df& a) {
  vec2df res(a.shape);
  double sum = 0;
  for (int i = 0; i < a.size; i++) {
    res.data[i] = (float) exp(a.data[i]);
    sum += res.data[i];
  } // for
  for (int i = 0; i < res.size; i++)
    res.data[i] /= sum;
  return res;
} // softmax()

vec2df vec2df::operator+(const vec2df& rhs) const {
  assert(this->shape == rhs.shape);
  vec2df res(this->shape);
  for (int i = 0; i < res.size; i++)
    res.data[i] = this->data[i] + rhs.data[i];
  return res;
} // operator+()

vec2df& vec2df::operator+=(const vec2df& rhs) {
  assert(this->shape == rhs.shape);
  for (int i = 0; i < this->size; i++)
    this->data[i] += rhs.data[i];
  return *this;
} // operator+=()

vec2df& vec2df::operator-=(const vec2df& rhs) {
  assert(this->shape == rhs.shape);
  for (int i = 0; i < this->size; i++)
    this->data[i] -= rhs.data[i];
  return *this;
} // operator-=()

vec2df vec2df::operator-(const vec2df& rhs) const {
  assert(this->shape == rhs.shape);
  vec2df res(this->shape);
  for (int i = 0; i < res.size; i++)
    res.data[i] = this->data[i] - rhs.data[i];
  return res;
} // operator-()

bool vec2df::operator==(const vec2df& rhs) const {
  float epsilon = 0.0001;
  if (this->shape != rhs.shape) 
    return false;
  for (int i = 0; i < this->size; i++) {
    if (std::fabs(this->data[i] - rhs.data[i]) >= epsilon) 
      return false;
  } // for
  return true;
} // operator==()

vec2df vec2df::operator/(const vec2df& rhs) const {
  assert(this->shape == rhs.shape);
  vec2df res(this->shape);
  for (int i = 0; i < res.size; i++)
    res.data[i] = this->data[i] / rhs.data[i];
  return res;
} // operator*()

vec2df vec2df::operator*(const vec2df& rhs) const {
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
