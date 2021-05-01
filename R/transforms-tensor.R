#' @export
transform_convert_image_dtype.torch_tensor <- function(img, dtype = torch::torch_float()) {

  if (img$dtype == dtype)
    return(img)

  if (img$is_floating_point()) {

    # float to float
    if (dtype$is_floating_point)
      return(img$to(dtype = dtype))

    # float to int
    if ((img$dtype == torch::torch_float32() &&
         (dtype == torch::torch_float32() || dtype == torch::torch_float64())) ||
        (img$dtype == torch::torch_float64() && dtype == torch::torch_int64())
    )
      runtime_error("The cast from {img$dtype} to {dtype} cannot be performed safely.")

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
      max <- torch::torch_iinfo(img$dtype)$max
      img <- img$to(dtype)

      return(img/max)
    }


    # int to int
    input_max <- torch::torch_iinfo(img$dtype)$max
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

  dtype <- img$dtype
  mean <- torch::torch_tensor(mean, dtype=dtype, device=img$device)
  std <- torch::torch_tensor(std, dtype=dtype, device=img$device)

  if ((std == 0)$any()$item()) {
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
transform_resize.torch_tensor <- function(img, size, interpolation = 2, max_size = NULL) {

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

  if (length(size) == 1) {

    if (w <= h) {
      short <- w
      long <- h
    } else {
      short <- h
      long <- w
    }

    requested_new_short <- size

    if (short == requested_new_short)
      return(img)

    new_short <- requested_new_short
    new_long <- as.integer(requested_new_short * long / short)

    if (!is.null(max_size))
      rlang::abort("Not implemented.")

    if (w <= h) {
      new_w <- new_short
      new_h <- new_long
    } else {
      new_w <- new_long
      new_h <- new_short
    }

  } else {
    new_w <- size[2]
    new_h <- size[1]
  }

  # make NCHW
  need_squeeze <- FALSE
  if (img$dim() < 4) {
    img <- img$unsqueeze(1)
    need_squeeze <- TRUE
  }

  mode <- interpolation_modes[interpolation == names(interpolation_modes)]
  mode <- unname(mode)

  out_dtype <- img$dtype
  need_cast <- FALSE
  if (!img$dtype == torch::torch_float32() &&
      !img$dtype == torch::torch_float64()) {
    need_cast <- TRUE
    img <- img$to(dtype = torch::torch_float32())
  }

  if (mode %in% c("bilinear", "bicubic"))
    align_corners <- FALSE
  else
    align_corners <- NULL


  img <- torch::nnf_interpolate(img, size = c(new_h, new_w), mode = mode,
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

  out_dtype <- img$dtype
  need_cast <- FALSE

  if (padding_mode != "constant" &&
      (img$dtype == torch::torch_float32() || img$dtype == torch::torch_float64())) {
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

  mean <- torch::torch_mean(transform_rgb_to_grayscale(img)$to(torch::torch_float()))

  blend(img, mean, contrast_factor)
}

#' @export
transform_adjust_hue.torch_tensor <- function(img, hue_factor) {

  if (hue_factor < 0.5 || hue_factor > 0.5)
    value_error("hue_factor must be between -0.5 and 0.5.")

  check_img(img)

  orig_dtype <- img$dtype
  if (img$dtype == torch::torch_uint8())
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


  blend(img, transform_rgb_to_grayscale(img), saturation_factor)
}

#' @export
transform_rotate.torch_tensor <- function(img, angle, resample = 0, expand = FALSE,
                                          center = NULL, fill = NULL) {

  center_f <- c(0.0, 0.0)
  if (!is.null(center)) {
    img_size <- get_image_size(img)
    # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
    center_f <- mapply(function(c, s) 1 * (c - s * 0.5), center, img_size)
  }


  # due to current incoherence of rotation angle direction between affine and
  # rotate implementations we need to set -angle.
  matrix <- get_inverse_affine_matrix(center_f, -angle, c(0.0, 0.0), 1.0, c(0.0, 0.0))

  rotate_impl(img, matrix=matrix, resample=resample, expand=expand, fill=fill)
}

#' @export
transform_affine.torch_tensor <- function(img, angle, translate, scale, shear,
                                          resample = 0, fillcolor = NULL) {

  if (!is.numeric(angle))
    value_error("Argument angle should be int or float")

  if (!length(translate) == 2)
    value_error("translate should be length 2")

  if (scale < 0)
    value_error("Argument scale should be positive")

  if (!is.numeric(shear))
    type_error("Shear should be either a single value or a sequence of 2 values")

  if (length(shear) == 1)
    shear <- c(shear, 0)

  img_size <- get_image_size(img)

  matrix <- get_inverse_affine_matrix(c(1, 1), angle, translate, scale, shear)
  affine_impl(img, matrix=matrix, resample=resample, fillcolor=fillcolor)
}

#' @export
transform_perspective.torch_tensor <- function(img, startpoints, endpoints, interpolation = 2,
                                               fill = NULL) {
  not_implemented_error("transform perspective is not implemented yet.")
}

#' @export
transform_rgb_to_grayscale.torch_tensor <- function(img) {
  check_img(img)
  (0.2989 * img[1] + 0.5870 * img[2] + 0.1140 * img[3])$to(img$dtype)
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
get_image_size.torch_tensor <- function(img) {
  check_img(img)

  rev(tail(img$size(), 2))
}

blend <- function(img1, img2, ratio) {

  if (img1$is_floating_point())
    bound <- 1
  else
    bound <- 255

  (ratio * img1 + (1 - ratio) * img2)$clamp(0, bound)$to(img1$dtype)
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

  mask <- i == torch::torch_arange(start= 0, end = 5)[,NULL,NULL]

  a1 <- torch::torch_stack(list(hsv[[3]], q, p, p, t, hsv[[3]]))
  a2 <- torch::torch_stack(list(t, hsv[[3]], hsv[[3]], q, p, p))
  a3 <- torch::torch_stack(list(p, p, t, hsv[[3]], hsv[[3]], q))
  a4 <- torch::torch_stack(list(a1, a2, a3))

  torch::torch_einsum("ijk, xijk -> xjk", list(mask$to(dtype = img$dtype), a4))
}

# https://stackoverflow.com/questions/32370485/convert-radians-to-degree-degree-to-radians
rad2deg <- function(rad) {(rad * 180) / (pi)}
deg2rad <- function(deg) {(deg * pi) / (180)}

# Helper method to compute inverse matrix for affine transformation
# As it is explained in PIL.Image.rotate
# We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
# where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
#       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
#       RSS is rotation with scale and shear matrix
#       RSS(a, s, (sx, sy)) =
#       = R(a) * S(s) * SHy(sy) * SHx(sx)
#       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
#         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
#         [ 0                    , 0                                      , 1 ]
#
# where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
# SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
#          [0, 1      ]              [-tan(s), 1]
#
# Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1
get_inverse_affine_matrix <- function(center, angle, translate, scale, shear) {

  rot = deg2rad(angle)
  sx <- deg2rad(shear[1])
  sy <- deg2rad(shear[2])

  cx <- center[1]
  cy <- center[2]

  tx <- translate[1]
  ty <- translate[2]

  # RSS without scaling
  a <- cos(rot - sy) / cos(sy)
  b = -cos(rot - sy) * tan(sx) / cos(sy) - sin(rot)
  c = sin(rot - sy) / cos(sy)
  d = -sin(rot - sy) * tan(sx) / cos(sy) + cos(rot)

  # Inverted rotation matrix with scale and shear
  # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
  matrix <- c(d, -b, 0.0, -c, a, 0.0) / scale

  # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
  matrix[3] = matrix[3] + matrix[1] * (-cx - tx) + matrix[2] * (-cy - ty)
  matrix[6] = matrix[6] + matrix[4] * (-cx - tx) + matrix[5] * (-cy - ty)

  # Apply center translation: C * RSS^-1 * C^-1 * T^-1
  matrix[3] = matrix[3] + cx
  matrix[6] = matrix[6] + cy

  matrix
}

assert_grid_transform_inputs <- function(img, matrix, resample, fillcolor,
                                         interpolation_modes, coeffs) {
  check_img(img)
}

apply_grid_transform <- function(img, grid, mode) {

  need_squeeze <- FALSE
  if (img$dim() < 4) {
    img <- img$unsqueeze(dim=1)
    need_squeeze <- TRUE
  }

  out_dtype <- img$dtype
  need_cast <- FALSE
  if (!img$dtype == torch::torch_float32() && !img$dtype == torch::torch_float64()) {
    need_cast <- TRUE
    img <- img$to(dtype = torch::torch_float32())
  }

  img <- torch::nnf_grid_sample(img, grid, mode=mode, padding_mode="zeros",
                                align_corners=FALSE)

  if (need_squeeze)
    img <- img$squeeze(dim=1)

  if (need_cast) {
    # it is better to round before cast
    img <- torch::torch_round(img)$to(dtype = out_dtype)
  }

  img
}

# https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
# AffineGridGenerator.cpp#L18
# Difference with AffineGridGenerator is that:
# 1) we normalize grid values after applying theta
# 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate
gen_affine_grid <- function(theta, w, h, ow, oh) {

  d = 0.5
  base_grid = torch::torch_empty(1, oh, ow, 3)
  base_grid[.., 1]$copy_(torch::torch_linspace(start = -ow * 0.5 + d, end = ow * 0.5 + d - 1,
                                               steps=ow))

  base_grid[.., 2]$copy_(torch::torch_linspace(start = -oh * 0.5 + d, end = oh * 0.5 + d - 1,
                                               steps=oh)$unsqueeze_(-1))
  base_grid[.., 3]$fill_(1)

  output_grid = base_grid$view(c(1, oh * ow, 3))$bmm(
    theta$transpose(2, 3) / torch::torch_tensor(c(0.5 * w, 0.5 * h))
  )

  output_grid$view(c(1, oh, ow, 2))
}

rotate_compute_output_size <- function(theta, w, h) {
  # Inspired of PIL implementation:
  # https://github.com/python-pillow/Pillow/blob/11de3318867e4398057373ee9f12dcb33db7335c/src/PIL/Image.py#L2054

  # pts are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
  pts <- torch::torch_tensor(rbind(
    c(-0.5 * w, -0.5 * h, 1.0),
    c(-0.5 * w, 0.5 * h, 1.0),
    c(0.5 * w, 0.5 * h, 1.0),
    c(0.5 * w, -0.5 * h, 1.0)
  ))
  new_pts <- pts$view(c(1, 4, 3))$bmm(theta$transpose(2, 3))$view(c(4, 2))
  min_vals <- new_pts$min(dim=1)[[1]]
  max_vals <- new_pts$max(dim=1)[[1]]

  # Truncate precision to 1e-4 to avoid ceil of Xe-15 to 1.0
  tol = 1e-4
  cmax = torch::torch_ceil((max_vals / tol)$trunc_() * tol)
  cmin = torch::torch_floor((min_vals / tol)$trunc_() * tol)
  size = cmax - cmin

  as.integer(c(size[1]$item(), size[2]$item()))
}

rotate_impl <- function(img, matrix, resample = 0, expand = FALSE, fill= NULL) {

  interpolation_modes <- c(
    "0" =  "nearest",
    "2" = "bilinear"
  )

  assert_grid_transform_inputs(img, matrix, resample, fill, interpolation_modes)
  theta <- torch::torch_tensor(matrix)$reshape(c(1, 2, 3))
  w <- tail(img$shape, 2)[2]
  h <- tail(img$shape, 1)[1]

  if (expand) {
    o_shape <- rotate_compute_output_size(theta, w, h)
    ow <- o_shape[1]
    oh <- o_shape[2]
  } else {
    ow <- w
    oh <- h
  }

  grid <- gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)
  mode <- interpolation_modes[as.character(resample)]

  apply_grid_transform(img, grid, mode)
}

affine_impl <- function(img, matrix, resample = 0, fillcolor = NULL) {

  interpolation_modes <- c(
    "0"= "nearest",
    "2"= "bilinear"
  )

  assert_grid_transform_inputs(img, matrix, resample, fillcolor, interpolation_modes)

  theta <- torch::torch_tensor(matrix, dtype=torch::torch_float())$reshape(c(1, 2, 3))
  shape <- img$shape
  grid <- gen_affine_grid(theta, w=rev(shape)[1], h=rev(shape)[2],
                         ow=rev(shape)[1], oh=rev(shape)[2])
  mode <- interpolation_modes[as.character(resample)]
  apply_grid_transform(img, grid, mode)
}

pad_symmetric <- function(img, padding) {

  in_sizes <- img$size()

  x_indices <- seq_len(tail(in_sizes, 1))
  left_indices <- rev(seq_len(padding[1]))
  right_indices <- -seq_len(padding[2])
  x_indices <- torch::torch_tensor(c(left_indices, x_indices, right_indices),
                                   dtype = torch::torch_long())


  y_indices <- seq_len(rev(in_sizes)[2])
  top_indices <- rev(seq_len(padding[3]))
  bottom_indices <- -seq_len(padding[4])
  y_indices <- torch::torch_tensor(c(top_indices, y_indices, bottom_indices),
                                   dtype = torch::torch_long())


  ndim <- length(dim(img))
  if (ndim == 3)
    img[, y_indices[, NULL], x_indices[, NULL]]
  else if (ndim == 4)
    img[,,y_indices[, NULL], x_indices[, NULL]]
  else
    runtime_error("Symmetric padding of N-D tensors are not supported yet")
}

#' @export
transform_adjust_gamma.torch_tensor <- function(img, gamma, gain = 1) {

  check_img(img)

  if (gamma < 0)
    value_error("gamma must be non-negative")

  result <- img
  dtype <- img$dtype

  if (!torch::torch_is_floating_point(img))
    result <- result/255.0

  result <- (gain * result ^ gamma)$clamp(0, 1)

  if (!result$dtype == dtype) {
    eps <- 1e-3
    result <- (255 + 1.0 - eps) * result
  }

  result <- result$to(dtype = dtype)
  result
}

get_perspective_coeffs <- function(startpoints, endpoints) {

  a_matrix <- torch::torch_zeros(2 * length(startpoints), 8,
                                 dtype = torch::torch_float())

  for (i in seq_along(startpoints)) {

    p1 <- endpoints[[i]]
    p2 <- startpoints[[i]]

    a_matrix[1 + 2*(i-1), ] <- torch::torch_tensor(c(p1[1], p1[2], 2, 1, 1, 1, -p2[1] * p1[1], -p2[1] * p1[2]))
    a_matrix[2*i,] <- torch::torch_tensor(c(1, 1, 1, p1[1], p1[2], 2, -p2[2] * p1[1], -p2[2] * p1[1]))

  }

  b_matrix <- torch::torch_tensor(unlist(startpoints), dtype = torch::torch_float())$view(8)
  res <- torch::torch_lstsq(b_matrix, a_matrix)[[1]]

  output <- torch::as_array(res$squeeze(1))

  output
}

