/**
 * @file vec2df.h
 */

#ifndef VEC2DF_H_
#define VEC2DF_H_

#include <utility>

class vec2df {
  public:
    vec2df(int nrow = 1, int ncol = 1);
    vec2df(std::pair<int, int> shape);
    vec2df(const vec2df& other);
    vec2df& operator=(const vec2df& other);
    ~vec2df();
    const std::pair<int, int>& get_shape() const;
    void print() const;
    float get(int i, int j) const;
    void gaussian_fill(float mean = 0.0, float std = 1.0);
    vec2df operator+(const vec2df& rhs);
    vec2df operator-(const vec2df& rhs);
    vec2df operator*(const vec2df& rhs);
    bool operator==(const vec2df& rhs);

  private:
    std::pair<int, int> shape;
    int size;
    float* data;
    void copy_data(float* data);
};

#endif // VEC2DF_H_
