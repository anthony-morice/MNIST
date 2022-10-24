#include <mnist.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <cassert>
#include <byteswap.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <cstdlib>
#include <ctime>
#include <numeric>

Mnist::Mnist(std::string imgs_path, std::string labels_path, int num_classes) {
  this->num_classes = num_classes;
  this->read_img_file(imgs_path);
  this->read_label_file(labels_path);
} // Mnist()

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
      cv::Mat img32f;
      img.convertTo(img32f, CV_32F);
      this->imgs.push_back(img32f);
    } // for
    ifs.close();
    this->num_images = num_images;
    this->image_dims = {num_rows, num_columns};
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

int Mnist::view_single(int i, std::string s) {
  auto im = this->imgs.at(i);
  cv::namedWindow("Img", cv::WINDOW_NORMAL);
  cv::imshow("Img", im);
  std::cout << "\n --Image " << i << "--" << std::endl;
  std::cout << "  Label: " << (int) this->labels.at(i) << std::endl;
  std::cout << "  Onehot: [ ";
  for (int val : this->get_onehot(i)) {
    std::cout << val << " ";
  } // for
  std::cout << "]" << std::endl;
  std::cout << s << std::endl;
  int wk = cv::waitKey(0);
  return wk;
} // view_single()

void Mnist::view(int i) {
  if (i < 0) { // show all images one by one until user quits
    for (int j = 0; j < (int) this->imgs.size(); j++) {
      int wk = this->view_single(j, "\nPress 'n' to view next image OR any other key to continue...\n");
      if (wk == 'n')
        std::cout << "... displaying next image" << std::endl;
      else 
        break;
    } // for
  } else
    this->view_single(i);
} // view_imgs()

std::vector<int> Mnist::get_onehot(int i) const {
  std::vector<int> v(this->num_classes, 0);
  v.at((int) this->labels.at(i)) = 1;
  return v;
} // get_onehot()

int Mnist::get_label(int i) const {
  return (int) this->labels.at(i);
} // get_label()

const cv::Mat& Mnist::get_image(int i) {
  return this->imgs.at(i);
} // get_training()

std::vector<std::vector<int>> Mnist::get_mini_batches(int batch_size) {
  std::srand(std::time(nullptr));
  std::vector<int> v(this->imgs.size());
  std::iota(v.begin(), v.end(), 0);
  // randomly permute vector 
  for (int i = v.size() - 1; i >= 1; i--) {
    int rn = std::rand() % i;
    std::swap(v.at(rn), v.at(i));
  } // for
  // construct mini batches
  std::vector<std::vector<int>> mbs;
  int i = 0;
  while (i + batch_size < (int) v.size()) {
    std::vector<int> mini_batch(batch_size);
    for (int j = 0; j < batch_size; j++)
      mini_batch[j] = v[i + j];
    i += batch_size;
    mbs.push_back(mini_batch);
  } // while
  // add a partial mini batch if training size not evenly divisible by batch size
  if (i < (int) v.size()) {
    std::vector<int> mini_batch;
    while (i < (int) v.size())
      mini_batch.push_back(v[i++]);
    mbs.push_back(mini_batch);
  } // if
  return mbs;
} // get_mini_batches()
