/**
 * @file mnist.h
 */

#ifndef MNIST_H_
#define MNIST_H_

#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>
#include <utility>

class Mnist {
  public:
    Mnist(std::string imgs_path, std::string labels_path, int num_classes = 10);
    const cv::Mat& get_image(int i);
    std::vector<int> get_onehot(int i);
    std::vector<std::vector<int>> get_mini_batches(int batch_size);
    void view(int i = -1);
    int num_classes;
    int num_images;
    std::pair<int, int> image_dims;

  private:
    std::vector<cv::Mat> imgs;
    std::vector<unsigned char> labels;
    void read_img_file(std::string path);
    void read_label_file(std::string path);
    int view_single(int i, std::string s = "\nPress any key to continue\n");
};

#endif // MNIST_H_
