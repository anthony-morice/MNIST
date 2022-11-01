find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../include)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

add_executable(vec2df_test vec2df_test.cpp)
target_link_libraries(vec2df_test
  PRIVATE
    GTest::gtest_main
    vec2df
)

add_executable(mlp_test mlp_test.cpp)
target_link_libraries(mlp_test
  PRIVATE
    GTest::gtest_main
    ${OpenCV_LIBS}
    mlp
    vec2df
)

include(GoogleTest)
gtest_discover_tests(vec2df_test)
gtest_discover_tests(mlp_test)
