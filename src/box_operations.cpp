#include <Rcpp.h>
using namespace Rcpp;

//' Calculate Box Areas (C++)
//'
//' Calculates bounding box areas using C++. Provides performance benefits
//' for very large batches of boxes when working with plain matrices.
//'
//' @param boxes Numeric matrix with N rows and 4 columns (x1, y1, x2, y2)
//' @return Numeric vector of areas
//'
//' @details
//' This is a low-level function. Most users should use \code{box_area_fast()}
//' or \code{box_area()} instead.
//'
//' @export
// [[Rcpp::export]]
NumericVector box_area_cpp(NumericMatrix boxes) {
  int n = boxes.nrow();
  
  if (boxes.ncol() != 4) {
    stop("Expected 4 columns for box coordinates");
  }
  
  NumericVector areas(n);
  
  for (int i = 0; i < n; i++) {
    double w = boxes(i, 2) - boxes(i, 0);
    double h = boxes(i, 3) - boxes(i, 1);
    areas[i] = (w < 0 || h < 0) ? 0.0 : w * h;
  }
  
  return areas;
}
