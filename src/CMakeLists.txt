find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(GSL REQUIRED)
include_directories(../include)
find_package(OpenMP)

add_library(mnist mnist.cpp)
target_link_libraries(mnist
  PUBLIC
    ${OpenCV_LIBS}
)

add_library(vec2df vec2df.cpp)
target_link_libraries(vec2df
  PRIVATE
    GSL::gsl
    GSL::gslcblas
)

add_library(mlp mlp.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mlp 
    PRIVATE 
      OpenMP::OpenMP_CXX
      mnist
      vec2df
  )
else()
  target_link_libraries(mlp
    mnist
    vec2df
  )
endif()

add_executable(mnist-mlp main.cpp)
target_link_libraries(mnist-mlp 
  mnist
  mlp
)
