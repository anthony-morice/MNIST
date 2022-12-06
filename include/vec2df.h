/**
 * @file vec2df.h
 */

#ifndef VEC2DF_H_
#define VEC2DF_H_

#include <utility>
#include <vector>

class vec2df {
  public:
    vec2df(int nrow = 1, int ncol = 1);
    vec2df(std::pair<int, int> shape);
    vec2df(int size, float* data);
    vec2df(std::vector<int> v);
    vec2df(std::vector<float> v);
    vec2df(const vec2df& other);
    vec2df& operator=(const vec2df& other);
    ~vec2df();
    const std::pair<int, int>& get_shape() const;
    void print() const;
    vec2df transpose() const;
    float get(int i, int j) const;
    float get(int i) const;
    float& at(int i, int j) const;
    float& at(int i) const;
    int argmax() const;
    int argmin() const;
    vec2df& scale(float f);
    vec2df& sqrt();
    vec2df& add(float f);
    void gaussian_fill(float mean = 0.0, float std = 1.0);
    void zeros_fill();
    static vec2df clip_min(const vec2df& a, float min_val);
    static vec2df clip_max(const vec2df& a, float max_val);
    static vec2df scaled_unit_step(const vec2df& a, float k);
    static vec2df element_multiply(const vec2df& a, const vec2df& b);
    static vec2df softmax(const vec2df& a);
    vec2df operator+(const vec2df& rhs) const;
    vec2df& operator+=(const vec2df& rhs);
    vec2df& operator-=(const vec2df& rhs);
    vec2df operator-(const vec2df& rhs) const;
    vec2df operator/(const vec2df& rhs) const;
    vec2df operator*(const vec2df& rhs) const;
    bool operator==(const vec2df& rhs) const;
    float* data;
    int size;
    std::pair<int, int> shape;
};

#endif // VEC2DF_H_
