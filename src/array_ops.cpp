#include <Rcpp.h>
using namespace Rcpp;


//' Array Operations (C++ implementation)
//'
//' Element-wise addition of two numeric vectors and matrices using C++.
//'
//' @param vec_a First numeric vector
//' @param vec_b Second numeric vector
//'
//' @return A numeric vector with element-wise sums
//'
//' @details
//' Both vectors must have the same length.
//' If lengths differ, the function will recycle elements.
//'
//' @examples
//' \dontrun{
//' #Vector addition with recycling
//' add_vectors(c(1, 2, 3), c(4, 5, 6))  # Returns c(5, 7, 9)
//'
//' #Matrix addition
//' mat1 <- matrix(1:4, nrow = 2)
//' mat2 <- matrix(5:8, nrow = 2)
//' add_matrices(mat1, mat2)
//' }
//' @name array_ops
//' @export
// [[Rcpp::export]]
NumericVector add_vectors(NumericVector vec_a, NumericVector vec_b) {
  int n_a = vec_a.size();
  int n_b = vec_b.size();
  int n = std::max(n_a, n_b);

  NumericVector result(n);

  for (int i = 0; i < n; i++) {
    result[i] = vec_a[i % n_a] + vec_b[i % n_b];
  }

  return result;
}


//' @param mat_a First numeric matrix
//' @param mat_b Second numeric matrix  
//'
//' @return A numeric matrix with element-wise sums
//'
//' @details
//' Both matrices must have the same dimensions.
//' Throws an error if dimensions don't match.
//'
//' @rdname array_ops
//' @export
// [[Rcpp::export]]
NumericMatrix add_matrices(NumericMatrix mat_a, NumericMatrix mat_b) {
  if (mat_a.nrow() != mat_b.nrow() || mat_a.ncol() != mat_b.ncol()) {
    stop("Matrices must have the same dimensions");
  }

  int n_rows = mat_a.nrow();
  int n_cols = mat_a.ncol();
  NumericMatrix result(n_rows, n_cols);

  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      result(i, j) = mat_a(i, j) + mat_b(i, j);
    }
  }

  return result;
}
