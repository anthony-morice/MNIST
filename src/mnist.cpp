#include "mnist.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <byteswap.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

void read_img_file(std::string path, std::vector<cv::Mat>& imgs) {
  int magic, num_images, num_rows, num_columns;
  std::cout << "Reading images" << std::endl;
  std::ifstream ifs(path, std::ifstream::binary);
  if (ifs) {
    ifs.read((char*) &magic, sizeof(int));
    ifs.read((char*) &num_images, sizeof(int));
    ifs.read((char*) &num_rows, sizeof(int));
    ifs.read((char*) &num_columns, sizeof(int));
    // convert all header info to little endian
    magic = __bswap_32(magic);
    num_images = __bswap_32(num_images); 
    num_rows = __bswap_32(num_rows);
    num_columns = __bswap_32(num_columns);
    assert(magic == 2051);
    /*
    std::cout << "num_images: " << num_images << std::endl;
    std::cout << "  rows: " << num_rows << std::endl;
    std::cout << "  rows: " << num_columns << std::endl;
    */
    for (int i = 0; i < num_images; i++) {
      cv::Mat img(num_rows, num_columns, CV_8UC1);
      ifs.read((char*) img.data, num_rows * num_columns); 
      imgs.push_back(img);
    } // for
    ifs.close();
    std::cout << "  img dimensions: (" << num_rows << " X " 
              << num_columns << ")" << std::endl;
    std::cout << "  imgs read: " << imgs.size() << std::endl;
  } else {
    std::cout << "Error: could not open image file" << std::endl;
    exit(1);
  } // else
} // read_img_file()

void read_label_file(std::string path, std::vector<unsigned char>& labels) {
  int magic, num_labels;
  std::cout << "Reading labels" << std::endl;
  std::ifstream ifs(path, std::ifstream::binary);
  if (ifs) {
    ifs.read((char*) &magic, sizeof(int));
    ifs.read((char*) &num_labels, sizeof(int));
    // convert all header info to little endian
    magic = __bswap_32(magic);
    num_labels = __bswap_32(num_labels); 
    assert(magic == 2049);
    unsigned char uchar;
    for (int i = 0; i < num_labels; i++) {
      ifs.read((char*) &uchar, sizeof(unsigned char)); 
      labels.push_back(uchar);
    } // for
    ifs.close();
    std::cout << "  labels read: " << labels.size() << std::endl;
  } else {
    std::cout << "Error: could not open labels file" << std::endl;
    exit(1);
  } // else
} // read_label_file()

Mnist::Mnist(std::string train_imgs, std::string train_labels,
      std::string test_imgs, std::string test_labels) {
  read_img_file(train_imgs, this->train_imgs);
  read_label_file(train_labels, this->train_labels);
  read_img_file(test_imgs, this->test_imgs);
  read_label_file(test_labels, this->test_labels);
  for (int i = 0; i < (int) this->train_imgs.size(); i++) {
    auto im = this->train_imgs[i];
    cv::namedWindow("Img", cv::WINDOW_NORMAL);
    cv::imshow("Img", im);
    std::cout << "Label: " << (int) this->train_labels[i] << std::endl;
    int wk = cv::waitKey(0);
    if (wk == 'n')
      std::cout << "next" << std::endl;
    else 
      break;
  } // for
  exit(0);
} // Mnist()

Mnist::~Mnist() {

} // ~Mnist()
