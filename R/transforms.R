#' Convert a `Magick Image` or `array` to tensor.
#'
#' @param pic (`Magick Image` or `array`): Image to be converted to tensor.
#'
#' @return Tensor: Converted image.
#'
#' @export
transform_to_tensor <- function(pic) {

  if (!(is_magick_image(pic) || is_array_image(pic)))
    type_error("pic should be Magick image or array (with 2 or 3 dims)")


  # handle array
  if (is_array_image(pic)) {

    if (length(dim(pic)) == 2)
      dim(pic) <- c(dim(pic), 1)

    img <- torch::torch_tensor(pic)$transpose(c(3, 1, 2))

    return(img)
  }

  # handle magick
  if (is_magick_image(pic)) {
    return(transform_magick_to_tensor(pic))
  }

}

#' Convert a `PIL Image` to a tensor of the same type.
#'
#' @param pic (PIL Image): Image to be converted to tensor.
#' @return Tensor: Converted image.
#'
#' @export
transform_magick_to_tensor <- function(pic) {

  if (!is_magick_image(pic))
    type_error("pic must be a magick image")

  img <- as.integer(magick::image_data(pic))
  img <- torch::torch_tensor(img)$permute(c(3,1,2))
  img <- img$to(dtype = torch::torch_float32())
  img <- img$contiguous()
  img <- img$div(255)

  img
}

#' Convert a tensor image to the given `dtype` and scale the values accordingly
#'
#' @param image (torch.Tensor): Image to be converted
#' @param dtype (torch.dtype): Desired data type of the output
#'
#' @return (torch.Tensor): Converted image
#'
#' @note
#' When converting from a smaller to a larger integer `dtype` the maximum values
#' are **not** mapped exactly. If converted back and forth, this mismatch has no
#' effect.
#'
#' @section Raises:
#'
#' runtime_error: When trying to cast `torch_float32` to `torch_int32` or
#' `torch_int64` as well as for trying to cast `torch_float64` to `torch_int64`.
#' These conversions might lead to overflow errors since the floating point
#' `dtype` cannot store consecutive integers over the whole range of the integer
#' `dtype`.
#'
#' @export
transform_convert_image_dtype <- function(image, dtype = torch::torch_float()) {

  if (image$dtype() == dtype)
    return(image)

  if (image$is_floating_point()) {

    # float to float
    if (dtype$is_floating_point)
      return(image$to(dtype = dtype))

    # float to int
    if ((image$dtype() == torch::torch_float32() &&
        (dtype == torch::torch_float32() || dtype == torch::torch_float64())) ||
        (image$dtype() == torch::torch_float64() && dtype == torch::torch_int64())
        )
      runtime_error("The cast from {image$dtype()} to {dtype} cannot be performed safely.")

    # For data in the range 0-1, (float * 255).to(uint) is only 255
    # when float is exactly 1.0.
    # `max + 1 - epsilon` provides more evenly distributed mapping of
    # ranges of floats to ints.
    eps <- 1e-3
    result <- image$mul(torch::torch_iinfo(dtype)$max + 1 - eps)
    result <- result$to(dtype = dtype)

    return(result)
  } else {
    # int to float

    if (dtype$is_floating_point) {
      max <- torch::torch_iinfo(image$dtype())$max
      image <- image$to(dtype)

      return(image/max)
    }


    # int to int
    input_max <- torch::torch_iinfo(image$dtype())$max
    output_max <- torch::torch_iinfo(dtype)$max

    if (input_max > output_max) {
      factor <- (input_max + 1) %/% (output_max + 1)
      image = image %/% factor
      return(image$to(dtype = dtype))
    } else {

      factor <- (output_max + 1) %/% (input_max + 1)
      image <- image$to(dtype = dtype)

      return(image * factor)
    }
  }
}

#' Normalize a tensor image with mean and standard deviation.
#'
#' @note
#' This transform acts out of place by default, i.e., it does not mutates the
#' input tensor.
#'
#' @param tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#' @param mean (sequence): Sequence of means for each channel.
#' @param std (sequence): Sequence of standard deviations for each channel.
#' @param inplace (bool,optional): Bool to make this operation inplace.
#'
#' @return Tensor: Normalized Tensor image.
#'
#' @export
transform_normalize <- function(tensor, mean, std, inplace = FALSE) {

  check_img(tensor)

  if (!inplace)
    tensor <- tensor$clone()

  dtype <- tensor$dtype()
  mean <- torch::torch_tensor(mean, dtype=dtype, device=tensor$device())
  std <- torch::torch_tensor(std, dtype=dtype, device=tensor$device())

  if (torch::as_array((std == 0)$any())) {
    value_error("std evaluated to zero after conversion to {dtype}, leading to division by zero.")
  }

  if (mean$dim() == 1)
    mean <- mean[,NULL,NULL]

  if (std$dim() == 1)
    std <- std[,NULL,NULL]

  tensor$sub_(mean)$div_(std)

  tensor
}


is_magick_image <- function(x) {
  inherits(x, "magick-image")
}

is_array_image <- function(x) {
  is.array(x) && (length(dim(x)) %in% c(2, 3))
}
