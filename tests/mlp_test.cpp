#include <gtest/gtest.h>
#include <mlp.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <vec2df.h>
#include <cstdlib>
#include <iostream>
#include <vector>

#define EPS 0.00001

::testing::AssertionResult NEAR_FLOAT_EQ(float a, float b, float eps) {
  if(std::fabs(a - b) < eps)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure()
      << a << " is not within " << eps << " of " << b;
} // NEAR_FLOAT_EQ()

class MLPTest : public ::testing::Test {
  public: 
    void SetUp() override {
      model = new MLP(20, 10, 5);
      model->save_weights("mlp-test_weights.xml");
      cv::FileStorage fs("mlp-test_weights.xml", cv::FileStorage::READ);
      cv::Mat w1;
      cv::Mat w2;
      cv::Mat b1;
      cv::Mat b2;
      fs["w1"] >> w1;
      fs["w2"] >> w2;
      fs["b1"] >> b1;
      fs["b2"] >> b2;
      fs.release();
      this->w1 = vec2df(w1.size[0] * w1.size[1], (float*) w1.data);
      this->w1.shape = {w1.size[0], w1.size[1]};
      this->w2 = vec2df(w2.size[0] * w2.size[1], (float*) w2.data);
      this->w2.shape = {w2.size[0], w2.size[1]};
      this->b1 = vec2df(b1.size[0] * b1.size[1], (float*) b1.data);
      this->b1.shape = {b1.size[0], b1.size[1]};
      this->b2 = vec2df(b2.size[0] * b2.size[1], (float*) b2.data);
      this->b2.shape = {b2.size[0], b2.size[1]};
      in1 = vec2df(1, 20);
      for (int i = 0; i < 20; i++)
        in1.at(i) = float(i + 1);
      y1 = vec2df(1, 10);
      y1.zeros_fill();
      y1.at(2) = 1.0f;
    } // SetUp()

    void TearDown() override {
      delete model;
      std::system("rm ./mlp-test_weights.xml");
    } // TearDown()

  protected:
    MLP* model;
    vec2df w1, w2, b1, b2;
    vec2df in1, y1;
}; // Vector2DTest

/*******************************************************************************
 * Test Cases
 ******************************************************************************/

TEST_F(MLPTest, ConstructorTests) {
  std::pair<int, int> pw1 = {20,10};
  std::pair<int, int> pw2 = {10,5};
  std::pair<int, int> pb1 = {1,10};
  std::pair<int, int> pb2 = {1,5};
  EXPECT_EQ(w1.get_shape(), pw1) << "MLP - incorrect w1 shape initialization ";
  EXPECT_EQ(w2.get_shape(), pw2) << "MLP - incorrect w2 shape initialization ";
  EXPECT_EQ(b1.get_shape(), pb1) << "MLP - incorrect b1 shape initialization ";
  EXPECT_EQ(b2.get_shape(), pb2) << "MLP - incorrect b2 shape initialization ";
} // ConstructorTests

TEST_F(MLPTest, MathTests) {
  std::vector<float> vfb = {
         0.0181,  0.0491,  0.1822, -0.2178,  0.1308, -0.0976,
        -0.1767,  0.2150, -0.0333,  0.0040};
  std::vector<std::vector<float>> vfw = {
        { 0.1008,  0.1208, -0.1734,  0.0689,  0.0174, -0.0513, -0.1306,  0.1414,
          0.1326, -0.1306},
        {-0.1689,  0.1627,  0.1139,  0.1115, -0.1353,  0.0678, -0.0111,  0.1516,
         -0.1568,  0.1635},
        {-0.0444, -0.1298, -0.1456,  0.2047, -0.0801,  0.2116, -0.0905, -0.1188,
          0.0061, -0.0775},
        {-0.2110, -0.1855,  0.1637,  0.1928,  0.1934, -0.0504, -0.0574, -0.0641,
          0.1604,  0.2090},
        { 0.1152, -0.1239,  0.1488, -0.1532, -0.0949,  0.1681,  0.1975, -0.1282,
          0.0908, -0.2089},
        { 0.1472, -0.1013,  0.0474,  0.1735, -0.0760,  0.0889,  0.0597,  0.1717,
         -0.1012,  0.0953},
        {-0.2105, -0.0450,  0.1613, -0.1947, -0.1207,  0.1904, -0.1985, -0.2089,
         -0.0655,  0.0931},
        { 0.1063, -0.0264, -0.0770,  0.1557, -0.1387,  0.1238,  0.1261, -0.1055,
         -0.1848,  0.1220},
        {-0.0384, -0.0730,  0.0605,  0.1844,  0.0595, -0.1301, -0.0737, -0.1673,
         -0.1568, -0.0714},
        { 0.2167,  0.1778,  0.1801,  0.1138,  0.2199,  0.2132,  0.2028, -0.1721,
         -0.0519, -0.1802},
        {-0.0004, -0.0978, -0.2153,  0.1152, -0.0573, -0.1768, -0.1435,  0.1289,
         -0.0137,  0.0661},
        { 0.1704, -0.1499, -0.0203, -0.0228,  0.1372, -0.0399,  0.0907, -0.1071,
          0.1532,  0.1677},
        { 0.0626,  0.1722,  0.1477,  0.2069,  0.0514, -0.1781,  0.1317, -0.1188,
          0.1350, -0.1783},
        { 0.0566, -0.0919, -0.1979, -0.1017, -0.0559,  0.1790,  0.0974,  0.1198,
         -0.0403, -0.0978},
        {-0.1300,  0.1388,  0.2153,  0.1920,  0.0489, -0.1410, -0.2213, -0.0536,
          0.1857, -0.0754},
        { 0.0593,  0.0557,  0.1559, -0.0205, -0.0463,  0.1886, -0.1896,  0.0481,
          0.0015, -0.0731},
        { 0.1210, -0.0755,  0.2123,  0.1911,  0.1180, -0.0224, -0.0680,  0.2059,
          0.1884, -0.0398},
        {-0.0376,  0.0914, -0.0700, -0.0731,  0.0862,  0.1509,  0.1132,  0.1946,
          0.1636, -0.2006},
        { 0.1336,  0.1124, -0.0261,  0.0140, -0.1477,  0.1658, -0.1770, -0.0290,
         -0.0470, -0.1622},
        { 0.1870, -0.0382,  0.0171,  0.0006,  0.1608,  0.0749,  0.0786,  0.1440,
         -0.1262, -0.0656}};
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 10; j++)
      w1.at(i,j) = vfw[i][j];
  } // for 
  for (int i = 0; i < 10; i++)
    b1.at(i) = vfb[i];
  std::vector<float> val = { 11.7650,   1.5165,   8.7983,  11.4356,   5.1236,  10.7327,  -3.1779,
            4.4677,   4.8422, -12.4120};
  vec2df a1 = MLP::fc(in1, w1, b1);
  for (int i = 0; i < 10; i++)
    NEAR_FLOAT_EQ(a1.at(i), val[i], EPS) << "MLP - incorrect fc res at " << i;
  vec2df f1 = MLP::relu(a1);
  val = {11.7650,  1.5165,  8.7983, 11.4356,  5.1236, 10.7327,  0.0000,  4.4677, 4.8422,  0.0000};
  for (int i = 0; i < 10; i++)
    NEAR_FLOAT_EQ(f1.at(i), val[i], EPS) << "MLP - incorrect relu res at " << i;
  auto [loss, dl_dy] = MLP::loss_cross_entropy_softmax(f1, y1);
  NEAR_FLOAT_EQ(loss, 3.7229, EPS);
  val = { 4.6948e-01,  1.6625e-05, -9.7584e-01,  3.3772e-01,  6.1273e-04,
           1.6722e-01,  3.6487e-06,  3.1802e-04,  4.6245e-04,  3.6487e-06 };
  for (int i = 0; i < 10; i++)
    NEAR_FLOAT_EQ(dl_dy.at(i), val[i], EPS) << "MLP - incorrect dl_dy val at " << i;
  vec2df dl_da1 = MLP::relu_backward(dl_dy, a1, f1);
  val = { 4.6948e-01,  1.6625e-05, -9.7584e-01,  3.3772e-01,  6.1273e-04,
           1.6722e-01,  0.0000e+00,  3.1802e-04,  4.6245e-04,  0.0000e+00};
  for (int i = 0; i < 10; i++)
    NEAR_FLOAT_EQ(dl_da1.at(i), val[i], EPS) << "MLP - incorrect dl_da1 val at " << i;
  auto [dl_dx, dl_dw1, dl_db1] = MLP::fc_backward(dl_da1, in1, w1, b1, a1);
  val = { 3.2864e+00,  1.1638e-04, -6.8309e+00,  2.3641e+00,  4.2891e-03,
           1.1705e+00,  0.0000e+00,  2.2261e-03,  3.2372e-03,  0.0000e+00};
  for (int i = 0; i < 10; i++)
    NEAR_FLOAT_EQ(dl_db1.at(i), val[i], EPS) << "MLP - incorrect dl_db1 val at " << i;
  std::vector<std::vector<float>> val_2d = 
				{{ 3.2864e+00,  6.5727e+00,  9.8591e+00,  1.3145e+01,  1.6432e+01,
          1.9718e+01,  2.3005e+01,  2.6291e+01,  2.9577e+01,  3.2864e+01,
          3.6150e+01,  3.9436e+01,  4.2723e+01,  4.6009e+01,  4.9295e+01,
          5.2582e+01,  5.5868e+01,  5.9154e+01,  6.2441e+01,  6.5727e+01},
        { 1.1638e-04,  2.3275e-04,  3.4913e-04,  4.6550e-04,  5.8188e-04,
          6.9826e-04,  8.1463e-04,  9.3101e-04,  1.0474e-03,  1.1638e-03,
          1.2801e-03,  1.3965e-03,  1.5129e-03,  1.6293e-03,  1.7456e-03,
          1.8620e-03,  1.9784e-03,  2.0948e-03,  2.2111e-03,  2.3275e-03},
        {-6.8309e+00, -1.3662e+01, -2.0493e+01, -2.7323e+01, -3.4154e+01,
         -4.0985e+01, -4.7816e+01, -5.4647e+01, -6.1478e+01, -6.8309e+01,
         -7.5139e+01, -8.1970e+01, -8.8801e+01, -9.5632e+01, -1.0246e+02,
         -1.0929e+02, -1.1612e+02, -1.2296e+02, -1.2979e+02, -1.3662e+02},
        { 2.3641e+00,  4.7281e+00,  7.0922e+00,  9.4563e+00,  1.1820e+01,
          1.4184e+01,  1.6548e+01,  1.8913e+01,  2.1277e+01,  2.3641e+01,
          2.6005e+01,  2.8369e+01,  3.0733e+01,  3.3097e+01,  3.5461e+01,
          3.7825e+01,  4.0189e+01,  4.2553e+01,  4.4917e+01,  4.7281e+01},
        { 4.2891e-03,  8.5783e-03,  1.2867e-02,  1.7157e-02,  2.1446e-02,
          2.5735e-02,  3.0024e-02,  3.4313e-02,  3.8602e-02,  4.2891e-02,
          4.7180e-02,  5.1470e-02,  5.5759e-02,  6.0048e-02,  6.4337e-02,
          6.8626e-02,  7.2915e-02,  7.7204e-02,  8.1493e-02,  8.5783e-02},
        { 1.1705e+00,  2.3410e+00,  3.5115e+00,  4.6820e+00,  5.8525e+00,
          7.0230e+00,  8.1935e+00,  9.3641e+00,  1.0535e+01,  1.1705e+01,
          1.2876e+01,  1.4046e+01,  1.5217e+01,  1.6387e+01,  1.7558e+01,
          1.8728e+01,  1.9899e+01,  2.1069e+01,  2.2240e+01,  2.3410e+01},
        { 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00},
        { 2.2261e-03,  4.4522e-03,  6.6783e-03,  8.9044e-03,  1.1131e-02,
          1.3357e-02,  1.5583e-02,  1.7809e-02,  2.0035e-02,  2.2261e-02,
          2.4487e-02,  2.6713e-02,  2.8939e-02,  3.1165e-02,  3.3392e-02,
          3.5618e-02,  3.7844e-02,  4.0070e-02,  4.2296e-02,  4.4522e-02},
        { 3.2372e-03,  6.4744e-03,  9.7115e-03,  1.2949e-02,  1.6186e-02,
          1.9423e-02,  2.2660e-02,  2.5897e-02,  2.9135e-02,  3.2372e-02,
          3.5609e-02,  3.8846e-02,  4.2083e-02,  4.5321e-02,  4.8558e-02,
          5.1795e-02,  5.5032e-02,  5.8269e-02,  6.1506e-02,  6.4744e-02},
        { 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00}};
  std::cout << "shape: " << dl_dw1.shape.first << ", " << dl_dw1.shape.second << std::endl;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 20; j++)
			NEAR_FLOAT_EQ(dl_dw1.at(i,j), val_2d[i][j], EPS) << "MLP - incorrect dl_dw1 val at " << i << ", " << j;
  } // for 
	val = { 2.3136e-01, -1.4151e-01,  2.2564e-01, -2.0193e-01, -1.1485e-01,
         9.6325e-02, -2.9033e-01,  1.9815e-01, -3.6583e-02,  1.0881e-04,
         2.1928e-01,  8.5510e-02, -7.4599e-02,  2.1528e-01, -2.2972e-01,
        -9.9658e-02, -8.9373e-02,  5.1363e-02,  1.2050e-01,  8.3887e-02};
  for (int i = 0; i < 20; i++)
    NEAR_FLOAT_EQ(dl_dx.at(i), val[i], EPS) << "MLP - incorrect dl_dx val at " << i;
} // MathTests

TEST_F(MLPTest, SmallBackPropTest) {
  std::vector<float> in1 = {1, 2, 3, 4, 5};
  std::vector<float> y1 = {0, 1, 0};
  std::vector<std::vector<float>> w_fc = 
    {{-0.3139,  0.1732, -0.2063},
     {-0.0120,  0.0743,  0.2526},
     { 0.4344, -0.3323, -0.2040},
     {-0.2966,  0.0645, -0.0329},
     { 0.0751,  0.3800,  0.2389}};
  std::vector<float> b_fc = {0.0022, -0.3555, 0.2325};
  vec2df in(in1);
  vec2df y(y1);
  vec2df w(5, 3);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
      w.at(i, j) = w_fc[i][j];
    } // for
  } // for
  vec2df b(b_fc);
  vec2df a1 = MLP::fc(in, w, b);
  vec2df f1 = MLP::relu(a1);
  auto [y_tilde, dl_dy] = MLP::loss_cross_entropy_softmax(f1, y);
  vec2df dl_da1 = MLP::relu_backward(dl_dy, a1, f1);
  auto [dl_dx, dl_dw1, dl_db1] = MLP::fc_backward(dl_da1, in, w, b, a1);
  std::cout << "dl_dy:" << std::endl;
  dl_dy.print();
  std::cout << "dl_da1:" << std::endl;
  dl_da1.print();
  std::cout << "dl_dw1:" << std::endl;
  dl_dw1.print();
  std::cout << "dl_db1:" << std::endl;
  dl_db1.print();
  std::cout << "dl_dx:" << std::endl;
  dl_dx.print();
} // SmallBackPropTest 
