#include <iostream>
#include <gsl/gsl_cblas.h>

#define ROWS_C 2
#define COLUMNS_C 2
#define ROWS_A 2
#define COLUMNS_A 3
#define ROWS_B 3
#define COLUMNS_B 2

int main() {
  float a[] = {0.1, 0.2, 0.3,
               0.4, 0.5, 0.6};
  float b[] = {0.1, 0.2,
               0.3, 0.4,
               0.5, 0.6};
  float c[4];
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              ROWS_C, COLUMNS_C, COLUMNS_A, 1.0,
              a, COLUMNS_A, b, COLUMNS_B, 0.0, c, COLUMNS_C);
  std::cout << "[ ";
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      std::cout << c[2 * i + j] << ", "; 
    } // for
    std::cout << std::endl << "  ";
  } // for
  std::cout << "]" << std::endl;
  return 0;
} // main()
