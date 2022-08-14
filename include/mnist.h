/**
 * @file mnist.h
 */

#ifndef MNIST_H_
#define MNIST_H_

#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>

class Mnist {
  public:
    Mnist(std::string train_imgs, std::string train_labels,
          std::string test_imgs, std::string test_labels);
    ~Mnist();
    
  private:
    std::vector<cv::Mat> train_imgs;
    std::vector<unsigned char> train_labels;
    std::vector<cv::Mat> test_imgs;
    std::vector<unsigned char> test_labels;
};

#endif // MNIST_H_
