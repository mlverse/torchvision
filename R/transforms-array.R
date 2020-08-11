
#' @export
transform_to_tensor.array <- function(img) {
  if (length(dim(img)) == 2)
    dim(img) <- c(dim(img), 1)

  torch::torch_tensor(img)$transpose(c(3, 1, 2))
}

#' @export
transform_to_tensor.matrix <- transform_to_tensor.array
