#include <gtest/gtest.h>
#include <vec2df.h>
#include <cmath>

class Vec2dfTest : public ::testing::Test {
  public: 
    void SetUp() override {
      v0 = vec2df();     // 1 x 1
      v1 = vec2df(10);   // 10 x 1
      v2 = vec2df(4,5);  // 4 x 5
      v2.zeros_fill();
      std::pair<int,int> p = {3,3};
      v3 = vec2df(p);  // 3 x 3
      std::vector<int> v_std = {1, 2, 3, 4, 5};
      v4 = vec2df(v_std); // 1 x 5
      v5 = v2;
      float* data = new float[11];
      for (int i = 0; i < 11; i++)
        data[i] = float(i);
      v6 = vec2df(11, data);
      delete [] data;
    } // SetUp()

  protected:
    vec2df v0, v1, v2, v3, v4, v5, v6;
}; // Vector2DTest

/*******************************************************************************
 * Test Cases
 ******************************************************************************/

TEST_F(Vec2dfTest, ConstructorTests) {
  EXPECT_EQ(v0.get_shape().first, 1) << "v0 - incorrect initialization shape";
  EXPECT_EQ(v0.get_shape().second, 1) << "v0 - incorrect initialization shape";
  EXPECT_EQ(v1.get_shape().first, 10) << "v1 - incorrect initialization shape";
  EXPECT_EQ(v1.get_shape().second, 1) << "v1 - incorrect initialization shape";
  EXPECT_EQ(v2.get_shape().first, 4) << "v2 - incorrect initialization shape";
  EXPECT_EQ(v2.get_shape().second, 5) << "v2 - incorrect initialization shape";
  EXPECT_EQ(v3.get_shape().first, 3) << "v3 - incorrect initialization shape";
  EXPECT_EQ(v3.get_shape().second, 3) << "v3 - incorrect initialization shape";
  EXPECT_EQ(v4.get_shape().first, 1) << "v4 - incorrect initialization shape";
  EXPECT_EQ(v4.get_shape().second, 5) << "v4 - incorrect initialization shape";
  EXPECT_EQ(v5.get_shape().first, v2.get_shape().first) << "v5 - incorrect initialization shape";
  EXPECT_EQ(v5.get_shape().second, v2.get_shape().second) << "v5 - incorrect initialization shape";
  EXPECT_EQ(v6.get_shape().first, 1) << "v6 - incorrect initialization shape";
  EXPECT_EQ(v6.get_shape().second, 11) << "v6 - incorrect initialization shape";
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 5; j++)
      EXPECT_FLOAT_EQ(v2.get(i,j), 0.0) << "v2 - incorrect data initialization at index " << i << ", " << j;
  } // for
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(v4.get(i), float(i + 1)) << "v4 - incorrect data initialization at index " << i;
    EXPECT_FLOAT_EQ(v4.get(0, i), float(i + 1)) << "v4 - incorrect data initialization at index " << i;
  } // for
  for (int i = 0; i < 11; i++) {
    EXPECT_FLOAT_EQ(v6.get(i), float(i)) << "v6 - incorrect data initialization at index " << i;
    EXPECT_FLOAT_EQ(v6.get(0, i), float(i)) << "v6 - incorrect data initialization at index " << i;
  } // for
} // ConstructorTests

TEST_F(Vec2dfTest, ModiferTests) {
  v4.scale(3.0);
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(v4.get(i), 3 * float(i + 1)) << "v4 - scale by 3 issue at index " << i;
  } // for
  v4.sqrt();
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(v4.get(i), sqrt(3 * float(i + 1))) << "v4 - sqrt issue at index " << i;
  } // for
  vec2df v6_copy = v6;
  v6.add(5.0);
  for (int i = 0; i < 11; i++) {
    EXPECT_FLOAT_EQ(v6.get(i), float(i) + 5.0) << "v6 - element-wise add 5 issue at index " << i;
  } // for
  v6.add(-5.0);
  for (int i = 0; i < 11; i++) {
    EXPECT_FLOAT_EQ(v6.get(i), float(i)) << "v6 - element-wise add -5 issue at index " << i;
    EXPECT_FLOAT_EQ(v6.get(i), v6_copy.get(i)) << "v6 - element-wise add -5 issue at index " << i;
  } // for
  v6 += v6_copy;
  for (int i = 0; i < 11; i++) {
    EXPECT_FLOAT_EQ(v6.get(i), 2 * float(i)) << "v6 - += issue at index " << i;
  } // for
  v6 -= v6_copy;
  for (int i = 0; i < 11; i++) {
    EXPECT_FLOAT_EQ(v6.get(i), v6_copy.get(i)) << "v6 - += issue at index " << i;
  } // for
  v6.at(3) = 2.0;
  for (int i = 0; i < 11; i++) {
    if (i == 3)
      EXPECT_FLOAT_EQ(v6.get(i), 2.0) << "v6 - 'at' issue at index " << i;
    else
      EXPECT_FLOAT_EQ(v6.get(i), float(i)) << "v6 - += issue at index " << i;
  } // for
  v2.at(3,2) = 5.5;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 5; j++) {
      if (i == 3 && j == 2)
        EXPECT_FLOAT_EQ(v2.get(i,j), 5.5) << "v2 - 'at' issue at index " << i << ", " << j;
      else
        EXPECT_FLOAT_EQ(v2.get(i,j), 0.0) << "v2 - 'at' issue at index " << i << ", " << j;
    } // for
  } // for
  v6.zeros_fill();
  v6_copy.zeros_fill();
  EXPECT_EQ(v6, v6_copy) << "v6 == v6_copy issue";
  v6_copy.gaussian_fill();
  EXPECT_FALSE(v6_copy == v2) << "v6_copy == v6 issue";
  vec2df v2_copy = v2;
  EXPECT_EQ(v2_copy, v2) << "v2_copy == v2 issue";
  v2_copy.at(2,2) = 1.111;
  EXPECT_FALSE(v2_copy == v2) << "v2_copy == v2 issue";
} // ModiferTests 

TEST_F(Vec2dfTest, AdditionalTests) {
  vec2df v4_copy = v4;
  vec2df v2_copy = v2;
  vec2df v2_trans = v2.transpose();
  EXPECT_EQ(v2, v2_copy);
  EXPECT_FALSE(v2 == v2_trans);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 5; j++) {
        EXPECT_FLOAT_EQ(v2.get(i,j), v2_trans.get(j,i)) << "v2 - transpose issue at index " << i << ", " << j;
    } // for
  } // for
  EXPECT_EQ(10, v6.argmax()) << "v6 - argmax issue";
  EXPECT_EQ(0, v6.argmin()) << "v6 - argmin issue";
  v6.at(3) = 20.2;
  v6.at(2) = -20.5;
  EXPECT_EQ(3, v6.argmax()) << "v6 - argmax issue";
  EXPECT_EQ(2, v6.argmin()) << "v6 - argmin issue";
  vec2df clip_min = vec2df::clip_min(v4, 2.2);
  vec2df clip_max = vec2df::clip_max(v4, 3.2);
  vec2df min_expected = v4_copy;
  min_expected.at(0) = 2.2;
  min_expected.at(1) = 2.2;
  vec2df max_expected = v4_copy;
  max_expected.at(3) = 3.2;
  max_expected.at(4) = 3.2;
  EXPECT_EQ(min_expected, clip_min) << "v4 - clip_min issue";
  EXPECT_FALSE(clip_min == v4) << "v4 - clip_min issue";
  EXPECT_EQ(max_expected, clip_max) << "v4 - clip_max issue"; 
  EXPECT_FALSE(clip_max == v4) << "v4 - clip_max issue"; 
  clip_min = vec2df::clip_min(v4, -1.0);
  clip_max = vec2df::clip_max(v4, 7.0);
  EXPECT_EQ(clip_min, v4) << "v4 - clip_min issue";
  EXPECT_EQ(clip_max, v4) << "v4 - clip_max issue";
  v4.zeros_fill();
  v4.add(3);
  vec2df res = vec2df::element_multiply(v4_copy, v4);
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(res.get(i), float(i + 1) * 3) << "v4(res) - element_multiply issue at index " << i;
  } // for
  vec2df soft = vec2df::softmax(v4_copy);
  std::vector<float> expected_soft = {0.01165623095604, 0.031684920796124, 0.086128544436269, 0.23412165725274, 0.63640864655883};
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(soft.get(i), expected_soft[i]) << "softmax issue at index " << i;
  } // for
  res = v4 + v4_copy;
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(v4_copy.get(i) + 3, res.get(i)) << "operator + issue at index " << i;
  } // for
  res = v4 - v4_copy;
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(3 - v4_copy.get(i), res.get(i)) << "operator - issue at index " << i;
  } // for
  res = v4_copy / v4;
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(v4_copy.get(i) / 3.0, res.get(i)) << "operator / issue at index " << i;
  } // for
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 5; j++)
      v2.at(i,j) = i * 5 + j;
  } // for
  vec2df v2_t = v2.transpose();
  res = v2 * v2_t;
  std::pair<int,int> p = {4,4};
  EXPECT_EQ(res.get_shape(), p); 
  vec2df expected_v(std::vector<int> {30, 80, 130, 180, 80, 255, 430, 605, 130,
                                      430, 730, 1030, 180, 605, 1030, 1455});
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      EXPECT_FLOAT_EQ(res.at(i,j), expected_v.at(i*4 + j));
  } // for
} // AdditionalTests()
