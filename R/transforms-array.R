
#' @export
transform_to_tensor.array <- function(img) {
  # HW to HWC format for grayscale img
  if (length(dim(img)) == 2)
    dim(img) <- c(dim(img), 1)
  dims <- dim(img)
  ndim <- length(dims)

  is_3d_chw <- (ndim == 3 && dims[1] <= 4 && dims[3] > 4)

  if (ndim < 3 || ndim > 4)
    value_error("Expected a 2D (grayscale), 3D (image) or 4D (batch) array.")

  # Support both CHW / BCHW (channel-first arrays) and HWC (default image arrays) .
  if (is_3d_chw || ndim == 4) {
    res <- torch::torch_tensor(img)
  } else {
    res <- torch::torch_tensor(img)$permute(c(3, 1, 2))
  }

  if (res$dtype == torch::torch_long())
    res <- res/255

  res
}

#' @export
transform_to_tensor.matrix <- transform_to_tensor.array

#' @export
transform_to_tensor.list <- function(img) {
  if (inherits(img[[1]], "array")) {
    ndim <- length(dim(img[[1]]))
    if (ndim > 3) {
      value_error("Expected a list of 2D or 3D arrays.")
    }
    torch::torch_stack(lapply(img, transform_to_tensor))
  } else if (inherits(img[[1]], "magick-image")) {
    torch::torch_stack(lapply(img, transform_to_tensor))
  } else {
    not_implemented_for_class(img[[1]])
  }

}
