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
    Mnist(std::string imgs_path, std::string labels_path, int num_classes = 10);
    void read_img_file(std::string path);
    void read_label_file(std::string path);
    std::vector<int> get_onehot(int i);
    std::vector<std::vector<int>> get_mini_batches(int batch_size);
    void view();

  private:
    int num_classes;
    std::vector<cv::Mat> imgs;
    std::vector<unsigned char> labels;
};

#endif // MNIST_H_
