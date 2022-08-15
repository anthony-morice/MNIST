#include "mnist.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <byteswap.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

void Mnist::read_img_file(std::string path) {
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
    for (int i = 0; i < num_images; i++) {
      cv::Mat img(num_rows, num_columns, CV_8UC1);
      ifs.read((char*) img.data, num_rows * num_columns); 
      this->imgs.push_back(img);
    } // for
    ifs.close();
    std::cout << "  img dimensions: (" << num_rows << " X " 
              << num_columns << ")" << std::endl;
    std::cout << "  imgs read: " << this->imgs.size() << std::endl;
  } else {
    std::cout << "Error: could not open image file" << std::endl;
    exit(1);
  } // else
} // read_img_file()

void Mnist::read_label_file(std::string path) {
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
      this->labels.push_back(uchar);
    } // for
    ifs.close();
    std::cout << "  labels read: " << this->labels.size() << std::endl;
  } else {
    std::cout << "Error: could not open labels file" << std::endl;
    exit(1);
  } // else
} // read_label_file()

Mnist::Mnist(std::string imgs_path, std::string labels_path, int num_classes) {
  this->num_classes = num_classes;
  this->read_img_file(imgs_path);
  this->read_label_file(labels_path);
} // Mnist()

void Mnist::view() {
  for (int i = 0; i < (int) this->imgs.size(); i++) {
    auto im = this->imgs[i];
    cv::namedWindow("Img", cv::WINDOW_NORMAL);
    cv::imshow("Img", im);
    std::cout << "\nLabel: " << (int) this->labels[i] << std::endl;
    std::cout << "Onehot: [ ";
    for (int val : this->get_onehot(i)) {
      std::cout << val << " ";
    } // for
    std::cout << "]" << std::endl;
    std::cout << "\nPress 'n' to view next image OR any other key to continue...\n" << std::endl;
    int wk = cv::waitKey(0);
    if (wk == 'n')
      std::cout << "next" << std::endl;
    else 
      break;
  } // for
} // view_imgs()

std::vector<int> Mnist::get_onehot(int i) {
  assert(i >= 0 && i < (int) this->labels.size());
  std::vector<int> v(this->num_classes, 0);
  v.at((int) this->labels[i]) = 1;
  return v;
} // get_onehot()
