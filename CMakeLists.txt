cmake_minimum_required(VERSION 3.9)
project(MNIST)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
