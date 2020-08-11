#' @export
`transform_to_tensor.magick-image` <- function(img) {
  img <- as.integer(magick::image_data(img))
  img <- torch::torch_tensor(img)$permute(c(3,1,2))
  img <- img$to(dtype = torch::torch_float32())
  img <- img$contiguous()
  img <- img$div(255)

  img
}
