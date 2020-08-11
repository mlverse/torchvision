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

#' @export
transform_five_crop.torch_tensor <- function(img, size) {

  check_img(img)

  if (!length(size) == 2)
    value_error("Please provide only 2 dimensions (h, w) for size.")

  image_width <- img$size(2)
  image_height <- img$size(3)

  crop_height <- size[1]
  crop_width <- size[2]

  if (crop_width > image_width || crop_height > image_height)
    value_error("Requested crop size is bigger than input size.")

  tl <- transform_crop(img, 1, 1, crop_width, crop_height)
  tr <- transform_crop(img, image_width - crop_width + 1, 1, image_width, crop_height)
  bl <- transform_crop(img, 1, image_height - crop_height + 1, crop_width, image_height)
  br <- transform_crop(img, image_width - crop_width + 1, image_height - crop_height + 1, image_width, image_height)
  center <- transform_center_crop(img, c(crop_height, crop_width))

  list(tl, tr, bl, br, center)
}

#' @export
transform_ten_crop.torch_tensor <- function(img, size, vertical_flip = FALSE) {

  check_img(img)

  if (!length(size) == 2)
    value_error("Please provide only 2 dimensions (h, w) for size.")

  first_five <- transform_five_crop(img, size)

  if (vertical_flip)
    img <- transform_vflip(img)
  else
    img <- transform_hflip(img)

  second_five <- transform_five_crop(img, size)

  c(first_five, second_five)
}

#' @export
transform_linear_transformation.torch_tensor <- function(img, transformation_matrix,
                                                         mean_vector) {
  flat_tensor <- img$view(1, -1) - mean_vector
  transformed_tensor <- torch::torch_mm(flat_tensor, transformation_matrix)

  transformed_tensor$view(img$size())
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

#' @export
transform_adjust_brightness.torch_tensor <- function(img, brightness_factor) {
  if (brightness_factor < 0)
    value_error("brightness factor is negative")

  check_img(img)

  blend(img, torch::torch_zeros_like(img), brightness_factor)
}

#' @export
transform_adjust_contrast.torch_tensor <- function(img, contrast_factor) {

  if (contrast_factor < 0)
    value_error("contrast must be positive")

  check_img(img)

  mean <- torch::torch_mean(tft_rgb_to_grayscale(img)$to(torch::torch_float()))

  blend(img, mean, contrast_factor)
}

#' @export
transform_adjust_hue.torch_tensor <- function(img, hue_factor) {

  if (hue_factor < 0.5 || hue_factor > 0.5)
    value_error("hue_factor must be between -0.5 and 0.5.")

  check_img(img)

  orig_dtype <- img$dtype()
  if (img$dtype() == torch::torch_uint8())
    img <- img$to(dtype = torch::torch_float32())/255

  img <-rgb2hsv(img)
  hsv <- img$unbind(1)
  hsv[[1]] <- hsv[[1]] + hue_factor
  hsv[[1]] <- hsv[[1]] %% 1
  img <- torch::torch_stack(hsv)
  img_hue_adj <- hsv2rgb(img)

  if (orig_dtype == torch::torch_uint8())
    img_hue_adj <- (img_hue_adj * 255.0)$to(dtype=orig_dtype)

  img_hue_adj
}

#' @export
transform_adjust_saturation.torch_tensor <- function(img, saturation_factor) {

  if (saturation_factor < 0)
    value_error("saturation factor must be positive.")

  check_img(img)


  blend(img, tft_rgb_to_grayscale(img), saturation_factor)
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

blend <- function(img1, img2, ratio) {

  if (img1$is_floating_point())
    bound <- 1
  else
    bound <- 255

  (ratio * img1 + (1 - ratio) * img2)$clamp(0, bound)$to(img1$dtype())
}

rgb2hsv <- function(img) {

  rgb <- img$unbind(1)
  r <- rgb[[1]]; g <- rgb[[2]]; b <- rgb[[3]]

  maxc <- torch::torch_max(img, dim=1)[[1]]
  minc <- torch::torch_min(img, dim=1)[[1]]

  # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
  # from happening in the results, because
  #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
  #   + H channel has division by `(maxc - minc)`.
  #
  # Instead of overwriting NaN afterwards, we just prevent it from occuring so
  # we don't need to deal with it in case we save the NaN in a buffer in
  # backprop, if it is ever supported, but it doesn't hurt to do so.
  eqc <- maxc == minc

  cr <- maxc - minc

  # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
  s <- cr / torch::torch_where(eqc, maxc$new_full(list(), 1), maxc)
  # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
  # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
  # would not matter what values `rc`, `gc`, and `bc` have here, and thus
  # replacing denominator with 1 when `eqc` is fine.
  cr_divisor <- torch::torch_where(eqc, maxc$new_full(list(), 1), cr)
  rc <- (maxc - r) / cr_divisor
  gc <- (maxc - g) / cr_divisor
  bc <- (maxc - b) / cr_divisor

  hr <- (maxc == r) * (bc - gc)
  hg <- ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
  hb <- ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
  h <- (hr + hg + hb)
  h <- torch::torch_fmod((h / 6.0 + 1.0), 1.0)
  torch::torch_stack(list(h, s, maxc))
}

hsv2rgb <- function(img) {

  hsv <- img$unbind(1)
  i <- torch::torch_floor(hsv[[1]] * 6)
  f <- (hsv[[1]] * 6) - 1
  i <- i$to(dtype = torch::torch_int32())

  p <- torch::torch_clamp((hsv[[3]] * (1 - hsv[[2]])), 0, 1)
  q <- torch::torch_clamp((hsv[[3]] * (1 - hsv[[2]] * f)), 0, 1)
  t <- torch::torch_clamp((hsv[[3]] * (1 - f)), 0, 1)
  i <- i %% 6

  mask <- i == torch::torch_arange(start= 0, end = 6)[,NULL,NULL]

  a1 <- torch::torch_stack(list(hsv[[3]], q, p, p, t, hsv[[3]]))
  a2 <- torch::torch_stack(list(t, hsv[[3]], hsv[[3]], q, p, p))
  a3 <- torch::torch_stack(list(p, p, t, hsv[[3]], hsv[[3]], q))
  a4 <- torch::torch_stack(list(a1, a2, a3))

  torch::torch_einsum("ijk, xijk -> xjk", list(mask$to(dtype = img$dtype()), a4))
}
