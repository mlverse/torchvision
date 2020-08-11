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
