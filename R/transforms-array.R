
#' @export
transform_to_tensor.array <- function(img) {

  if (length(dim(img)) == 2)
    dim(img) <- c(dim(img), 1)

  res <- torch::torch_tensor(img)$permute(c(3, 1, 2))

  if (res$dtype == torch::torch_long())
    res <- res/255

  res
}

#' @export
transform_to_tensor.matrix <- transform_to_tensor.array
