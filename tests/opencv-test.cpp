#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "USAGE:\n ./a.out <path-to-img>" << std::endl; 
    return 1;
  } // if
  std::string img_path = argv[1];
  cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR); 
  if(img.empty()) {
    std::cout << "Could not read the image: " << img_path << std::endl;
    return 1;
  } // if
  cv::namedWindow("Img", cv::WINDOW_NORMAL);
  cv::imshow("Img", img);
  while (true) {
    std::cout << "Press 'q' to quit" << std::endl;
    int wk = cv::waitKey(0);
    if (wk == 'q')
      break;
  } // while
  return 0;
} // main()
