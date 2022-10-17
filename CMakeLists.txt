cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
project(MNIST)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(GSL REQUIRED)
include_directories(include)
# test-cblas
add_executable(test-cblas tests/cblas-test.cpp)
target_link_libraries(test-cblas GSL::gslcblas)
# test-opencv
add_executable(test-opencv tests/opencv-test.cpp)
target_link_libraries(test-opencv ${OpenCV_LIBS})
# test-mnist-read
add_executable(test-mnist-read tests/mnist-read-test.cpp src/mnist.cpp)
target_link_libraries(test-mnist-read ${OpenCV_LIBS})
# test-mlp
add_executable(test-mlp tests/mlp-test.cpp src/mnist.cpp src/mlp.cpp src/vec2df.cpp)
target_link_libraries(test-mlp ${OpenCV_LIBS} GSL::gslcblas)
# test-mlp
add_executable(mnist-mlp src/main.cpp src/mnist.cpp src/mlp.cpp src/vec2df.cpp)
target_link_libraries(mnist-mlp ${OpenCV_LIBS} GSL::gslcblas)
