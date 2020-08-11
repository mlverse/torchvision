#' @export
transform_convert_image_dtype.torch_tensor <- function(img, dtype = torch::torch_float()) {

  if (img$dtype() == dtype)
    return(img)

  if (img$is_floating_point()) {

    # float to float
    if (dtype$is_floating_point)
      return(img$to(dtype = dtype))

    # float to int
    if ((img$dtype() == torch::torch_float32() &&
         (dtype == torch::torch_float32() || dtype == torch::torch_float64())) ||
        (img$dtype() == torch::torch_float64() && dtype == torch::torch_int64())
    )
      runtime_error("The cast from {img$dtype()} to {dtype} cannot be performed safely.")

    # For data in the range 0-1, (float * 255).to(uint) is only 255
    # when float is exactly 1.0.
    # `max + 1 - epsilon` provides more evenly distributed mapping of
    # ranges of floats to ints.
    eps <- 1e-3
    result <- img$mul(torch::torch_iinfo(dtype)$max + 1 - eps)
    result <- result$to(dtype = dtype)

    return(result)
  } else {
    # int to float

    if (dtype$is_floating_point) {
      max <- torch::torch_iinfo(img$dtype())$max
      img <- img$to(dtype)

      return(img/max)
    }


    # int to int
    input_max <- torch::torch_iinfo(img$dtype())$max
    output_max <- torch::torch_iinfo(dtype)$max

    if (input_max > output_max) {
      factor <- (input_max + 1) %/% (output_max + 1)
      img = img %/% factor
      return(img$to(dtype = dtype))
    } else {

      factor <- (output_max + 1) %/% (input_max + 1)
      img <- img$to(dtype = dtype)

      return(img * factor)
    }
  }

}

#' @export
transform_normalize.torch_tensor <- function(img, mean, std, inplace = FALSE) {

  check_img(img)

  if (!inplace)
    img <- img$clone()

  dtype <- img$dtype()
  mean <- torch::torch_tensor(mean, dtype=dtype, device=tensor$device())
  std <- torch::torch_tensor(std, dtype=dtype, device=tensor$device())

  if (torch::as_array((std == 0)$any())) {
    value_error("std evaluated to zero after conversion to {dtype}, leading to division by zero.")
  }

  if (mean$dim() == 1)
    mean <- mean[,NULL,NULL]

  if (std$dim() == 1)
    std <- std[,NULL,NULL]

  img$sub_(mean)$div_(std)

  img
}

#' @export
transform_resize.torch_tensor <- function(img, size, interpolation = 2) {

  check_img(img)

  interpolation_modes <- c(
    "0" = "nearest",
    "2" = "bilinear",
    "3" = "bicubic"
  )

  if (!interpolation %in% names(interpolation_modes))
    value_error("This interpolation mode is unsupported with Tensor input")

  if (!length(size) %in% c(1,2) || !is.numeric(size))
    value_error("Size must be a numeric vector of length 1 or 2.")

  wh <- get_image_size(img)
  w <- wh[1]
  h <- wh[2]

  if (length(size) == 1)
    size_w <- size_h <- size
  else if (length(size) == 2) {
    size_w <- size[2]
    size_h <- size[1]
  }

  if (length(size) == 1) {

    if (w < h)
      size_h <- as.integer(size_w * h / w)
    else
      size_w <- as.integer(size_h * w / h)

  }

  if ((w <= h  && w == size_w) || (h <= w && h == size_h))
    return(img)

  # make NCHW
  need_squeeze <- FALSE
  if (img$dim() < 4) {
    img <- img$unsqueeze(1)
    need_squeeze <- TRUE
  }

  mode <- interpolation_modes[interpolation == names(interpolation_modes)]
  mode <- unname(mode)

  out_dtype <- img$dtype()
  need_cast <- FALSE
  if (!img$dtype() == torch::torch_float32() &&
      !img$dtype() == torch::torch_float64()) {
    need_cast <- TRUE
    img <- img$to(dtype = torch::torch_float32())
  }

  if (mode %in% c("bilinear", "bicubic"))
    align_corners <- FALSE
  else
    align_corners <- NULL


  img <- torch::nnf_interpolate(img, size = c(size_h, size_w), mode = mode,
                                align_corners = align_corners)

  if (need_squeeze)
    img <- img$squeeze(1)

  if (need_cast) {
    if (mode == "bicubic")
      img <- img$clamp(min = 0, max = 255)

    img <- img$to(out_dtype)
  }

  img
}

#' @export
transform_pad.torch_tensor <- function(img, padding, fill = 0, padding_mode = "constant") {

  check_img(img)

  if (!length(padding) %in% c(1,2,4) || !is.numeric(padding))
    value_error("Padding must be an int or a 1, 2, or 4 element numeric vector")

  if (!padding_mode %in% c("constant", "edge", "reflect", "symmetric"))
    value_error("Padding mode should be either constant, edge, reflect or symmetric")


  if (length(padding) == 1)
    pad_left <- pad_right <- pad_top <- pad_bottom <- padding
  else if (length(padding) == 2) {
    pad_left <- pad_right <- padding[1]
    pad_top <- pad_bottom <- padding[2]
  } else if (length(padding == 4)) {
    pad_left <- padding[1]
    pad_right <- padding[2]
    pad_top <- padding[3]
    pad_bottom <- padding[4]
  }

  p <- c(pad_left, pad_right, pad_top, pad_bottom)

  if (padding_mode == "edge")
    padding_mode <- "replicate"
  else if (padding_mode == "symmetric") {
    if (any(p < 0))
      value_error("Padding can not be negative for symmetric padding_mode")

    return(pad_symmetric(img, p))
  }

  need_squeeze <- FALSE

  if (img$dim() < 4) {
    img <- img$unsqueeze(1)
    need_squeeze <- TRUE
  }

  out_dtype <- img$dtype()
  need_cast <- FALSE

  if (padding_mode != "constant" &&
      (img$dtype() == torch::torch_float32() || img$dtype() == torch::torch_float64())) {
    need_cast <- TRUE
    img <- img$to(torch::torch_float32())
  }

  img <- torch::nnf_pad(img, p, mode = padding_mode, value = as.numeric(fill))

  if (need_squeeze)
    img <- img$squeeze(dim = 1)

  if (need_cast)
    img <- img$to(dtype = out_dtype)

  img
}


# Other methods -----------------------------------------------------------

#' @export
transform_crop.torch_tensor <- function(img, top, left, height, width) {
  check_img(img)

  img[.., top:(top + height - 1), left:(left + width - 1)]
}

#' @export
transform_hflip.torch_tensor <- function(img) {
  check_img(img)
  img$flip(-1)
}

#' @export
transform_vflip.torch_tensor <- function(img) {
  check_img(img)
  img$flip(-2)
}

# Helpers -----------------------------------------------------------------

is_tensor_image <- function(x) {
  x$dim() >= 2
}

check_img <- function(x) {
  if (!is_tensor_image(x))
    type_error("tensor is not a torch image.")
}

#' @importFrom utils tail
get_image_size <- function(img) {
  check_img(img)

  tail(img$size(), 2)
}
