#' @useDynLib torchvision, .registration = TRUE
#' @importFrom Rcpp evalCpp
NULL

#' Box Area Calculation (Rcpp demonstration)
#'
#' Alternative C++ implementation of box area calculation. This demonstrates
#' Rcpp integration and provides a performance boost for very large batches
#' (N > 100k boxes) when working with plain matrices instead of torch tensors.
#'
#' @param boxes Matrix (N x 4) with box coordinates (x1, y1, x2, y2), or
#'   torch tensor (will be converted to matrix)
#' @return Vector of areas (or torch tensor if input was torch tensor)
#'
#' @details
#' For most use cases, the standard \code{\link{box_area}} function is recommended.
#' This C++ version is useful when:
#' \itemize{
#'   \item Working with very large matrices (N > 100k)
#'   \item Input is already a plain R matrix
#'   \item Avoiding torch overhead is important
#' }
#'
#' @examples
#' \dontrun{
#' # For normal use, prefer box_area()
#' boxes_tensor <- torch_tensor(matrix(c(0,0,10,10), ncol=4))
#' box_area(boxes_tensor)
#'
#' # This function is faster for large plain matrices
#' boxes_matrix <- matrix(runif(1e6 * 4), ncol=4)
#' box_area_fast(boxes_matrix)
#' }
#'
#' @seealso \code{\link{box_area}}
#' @export
box_area_fast <- function(boxes) {
  is_torch <- inherits(boxes, "torch_tensor")
  
  if (is_torch) {
    boxes <- as.matrix(boxes$cpu())
  } else if (is.data.frame(boxes)) {
    boxes <- as.matrix(boxes)
  } else if (!is.matrix(boxes)) {
    stop("boxes must be a matrix, data.frame, or torch tensor")
  }
  
  areas <- box_area_cpp(boxes)
  
  if (is_torch && requireNamespace("torch", quietly = TRUE)) {
    areas <- torch::torch_tensor(areas, dtype = torch::torch_float())
  }
  
  areas
}
