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

include(GoogleTest)
gtest_discover_tests(vec2df_test)
