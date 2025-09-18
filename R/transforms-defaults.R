#' @export
transform_to_tensor.default <- function(img) {
  not_implemented_for_class(img)
}

#' @export
transform_convert_image_dtype.default <- function(img, dtype) {
  not_implemented_for_class(img)
}


#' @export
transform_normalize.default <- function(img, mean, std, inplace = FALSE) {
  not_implemented_for_class(img)
}

#' @export
transform_resize.default <- function(img, size, interpolation) {
  not_implemented_for_class(img)
}

#' @export
transform_center_crop.default <- function(img, size) {

  output_size <- size

  if (length(size) == 1)
    output_size <- rep(size, 2)

  output_size <- as.integer(output_size)

  size <- get_image_size(img)

  image_height <- size[2]
  image_width <- size[1]

  crop_height <- output_size[1]
  crop_width <- output_size[2]

  if (crop_width > image_width || crop_height > image_height) {

    padding_ltrb <- c(
      if (crop_width > image_width) (crop_width - image_width) %/% 2  else 0,
      if (crop_width > image_width) (crop_width - image_width + 1) %/% 2  else 0,
      if (crop_height > image_height) (crop_height - image_height) %/% 2  else 0,
      if (crop_height > image_height) (crop_height - image_height + 1) %/% 2  else 0
    )

    img <- transform_pad(img, padding_ltrb, fill = 0)  # PIL uses fill value 0

    size <- get_image_size(img)
    image_height <- size[1]
    image_width <- size[2]

    if (crop_width == image_width && crop_height == image_height) return(img)
  }

  crop_top <- as.integer((image_height - crop_height) / 2)
  crop_left <- as.integer((image_width - crop_width) / 2)

  # if either of these is 0, we will lose a pixel in transform_crop
  if (crop_top == 0) crop_top <- 1
  if (crop_left == 0) crop_left <- 1

  transform_crop(img, crop_top, crop_left, crop_height, crop_width)
}

#' @export
transform_pad.default <- function(img, padding, fill = 0, padding_mode = "constant") {
  not_implemented_for_class(img)
}

#' @export
transform_random_apply.default <- function(img, transforms, p = 0.5) {

  if (p < stats::runif(1))
    return(img)

  for (tf in transforms) {
    img <- tf(img)
  }

  img
}

#' @export
transform_random_choice.default <- function(img, transforms) {
  i <- sample.int(length(transforms), 1)
  transforms[[i]](img)
}

#' @export
transform_random_order.default <- function(img, transforms) {
  i <- sample.int(length(transforms), size = length(transforms))
  transforms <- transforms[i]
  for (tf in transforms) {
    img <- tf(img)
  }
  img
}

get_random_crop_params <- function(img, output_size) {

  img_size <- get_image_size(img)
  w <- img_size[1]; h <- img_size[2]
  th <- output_size[1]; tw <- output_size[2]

  if (w == tw && h == th)
    return(c(0, 0, h, w))

  i <- as.integer(torch::torch_randint(1, h - th + 1, size=1))
  j <- as.integer(torch::torch_randint(1, w - tw + 1, size=1))

  c(i, j, th, tw)
}

#' @export
transform_random_crop.default <- function(img, size, padding=NULL, pad_if_needed=FALSE,
                                          fill=0, padding_mode="constant") {


  if (length(size) == 1)
    size <- c(size, size)

  size <- as.integer(size)


  if (!is.null(padding))
    img <- transform_pad(img, padding, fill, padding_mode)

  img_size <- get_image_size(img)
  width <- img_size[1]; height <- img_size[2]

  # pad the width if needed
  if (pad_if_needed && width < size[2]) {
    padding <- c(size[2] - width, 0)
    img <- transform_pad(img, padding, fill, padding_mode)
  }

  # pad the height if needed
  if (pad_if_needed && height < size[1]) {
    padding <- c(0, size[1] - height)
    img <- transform_pad(img, padding, fill, padding_mode)
  }

  params <- get_random_crop_params(img, size)

  transform_crop(img, params[1], params[2], params[3], params[4])
}

#' @export
transform_random_horizontal_flip.default <- function(img, p = 0.5) {

  if (stats::runif(1) < p)
    img <- transform_hflip(img)

  img
}


#' @export
transform_random_vertical_flip.default <- function(img, p = 0.5) {

  if (stats::runif(1) < p)
    img <- transform_vflip(img)

  img

}

get_random_resized_crop_params <- function(img, scale, ratio) {

  img_size <- get_image_size(img)
  width <- img_size[1]; height <- img_size[2]

  area <- height * width

  for (i in 1:10) {
    target_area <- as.numeric(area * torch::torch_empty(1)$uniform_(scale[1], scale[2]))
    log_ratio <- torch::torch_log(torch::torch_tensor(ratio))
    aspect_ratio <-  as.numeric(torch::torch_exp(
      torch::torch_empty(1)$uniform_(as.numeric(log_ratio[1]), as.numeric(log_ratio[2]))
    ))

    w <- as.integer(round(sqrt(target_area * aspect_ratio)))
    h <- as.integer(round(sqrt(target_area / aspect_ratio)))

    if (1 < w && w <= width && 1 < h && h <= height) {

      i <- sample.int(height - h + 1, size=1)
      j <- sample.int(width - w + 1, size=1)

      return(c(i, j, h, w))
    }

  }

  # Fallback to central crop
  in_ratio <- width / height
  if (in_ratio < min(ratio)) {
    w <- width
    h <- as.integer(round(w / min(ratio)))
  } else if (in_ratio > max(ratio)) {
    h <-  height
    w <- as.integer(round(h * max(ratio)))
  } else {
    w <- width
    h <- height
  }
  i <- (height - h) %/% 2
  j <- (width - w) %/% 2

  c(i, j, h, w)
}

#' @export
transform_random_resized_crop.default <- function(img, size, scale=c(0.08, 1.0),
                                                  ratio=c(3. / 4., 4. / 3.),
                                                  interpolation=2) {


  params <- get_random_resized_crop_params(img, scale, ratio)

  transform_resized_crop(img, params[1], params[2], params[3], params[4], size,
                        interpolation)
}

#' @export
transform_five_crop.default <- function(img, size) {
  not_implemented_for_class(img)
}

#' @export
transform_ten_crop.default <- function(img, size, vertical_flip = FALSE) {
  not_implemented_for_class(img)
}

#' @export
transform_linear_transformation.default <- function(img, transformation_matrix,
                                                    mean_vector) {
  not_implemented_for_class(img)
}

get_color_jitter_params <- function(brightness, contrast, saturation, hue) {

  transforms <- list()

  if (!is.null(brightness)) {
    brightness_factor <- stats::runif(1, min = brightness[1], max = brightness[2])
    transforms <- append(
      transforms,
      list(function(img) transform_adjust_brightness(img, brightness_factor))
    )
  }

  if (!is.null(contrast)) {
    contrast_factor <- stats::runif(1, contrast[1], contrast[2])
    transforms <- append(
      transforms,
      list(function(img) transform_adjust_contrast(img, contrast_factor))
    )
  }

  if (!is.null(saturation)) {
    saturation_factor <- stats::runif(1, saturation[1], saturation[2])
    transforms <- append(
      transforms,
      list(function(img) transform_adjust_saturation(img, saturation_factor))
    )
  }

  if (!is.null(hue)) {
    hue_factor <- stats::runif(1, hue[1], hue[2])
    transforms <- append(
      transforms,
      list(function(img) transform_adjust_hue(img, hue_factor))
    )
  }

  # shuffle
  i <- sample.int(length(transforms), length(transforms))
  transforms <- transforms[i]

  function(img) {
    for (tf in transforms) {
      img <- tf(img)
    }
    img
  }
}

check_color_jitter_input <- function(value, center = 1, bound = c(0, Inf),
                                     clip_first_on_zero = TRUE) {

  if (length(value) == 1) {

    if (value < 0)
      value_error("must be positive if a single number")

    value <- c(center - value, center + value)

    if (clip_first_on_zero)
      value[1] <- max(value[1], 0.0)

  } else if (length(value) == 2) {

    if (value[1] < bound[1] || value[2] > bound[2])
      value_error("out of bounds.")

  }

  # if value is 0 or (1., 1.) for brightness/contrast/saturation
  # or (0., 0.) for hue, do nothing
  if (value[1] == value[2] && value[2] == center) {
    value <- NULL
  }

  value
}

#' @export
transform_color_jitter.default <- function(img, brightness=0, contrast=0,
                                           saturation=0, hue=0) {

  brightness <- check_color_jitter_input(brightness)
  contrast <- check_color_jitter_input(contrast)
  saturation <- check_color_jitter_input(saturation)
  hue <- check_color_jitter_input(hue, center=0, bound=c(-0.5, 0.5),
                               clip_first_on_zero=FALSE)

  tf <- get_color_jitter_params(brightness, contrast, saturation, hue)
  tf(img)
}

get_random_rotation_params <- function(degrees) {
  as.numeric(torch::torch_empty(1)$uniform_(degrees[1], degrees[2]))
}

#' @export
transform_random_rotation.default <- function(img, degrees, resample=FALSE,
                                              expand=FALSE, center=NULL, fill=NULL) {

  if (length(degrees) == 1) {

    if (degrees < 0)
      value_error("degrees must be positive if it's a single value")

    degrees <- c(-degrees, degrees)

  } else if (length(degrees) != 2) {
    value_error("degrees must be length 1 or 2")
  }

  angle <- get_random_rotation_params(degrees)
  transform_rotate(img, angle, resample, expand, center, fill)
}


get_random_affine_params <- function(degrees,
                                     translate,
                                     scale_ranges,
                                     shears,
                                     img_size) {

  angle <- as.numeric(torch::torch_empty(1)$uniform_(degrees[1], degrees[2]))
  if (!is.null(translate)) {
    max_dx <- as.numeric(translate[1] * img_size[1])
    max_dy <- as.numeric(translate[2] * img_size[2])
    tx <- as.integer(round(as.numeric(torch::torch_empty(1)$uniform_(-max_dx, max_dx))))
    ty <- as.integer(round(as.numeric(torch::torch_empty(1)$uniform_(-max_dy, max_dy))))
    translations <- c(tx, ty)
  } else {
    translations <- c(0,0)
  }

  if (!is.null(scale_ranges)) {
    scale <- as.numeric(torch::torch_empty(1)$uniform_(scale_ranges[1], scale_ranges[2]))
  } else {
    scale <- 1
  }

  shear_x <- shear_y <- 0.0

  if (!is.null(shears))  {
    shear_x <- as.numeric(torch::torch_empty(1)$uniform_(shears[1], shears[2]))
    if (length(shears) == 4)
      shear_y <- as.numeric(torch::torch_empty(1)$uniform_(shears[3], shears[4]))
  }

  shear <- c(shear_x, shear_y)

  list(angle, translations, scale, shear)
}

#' @export
transform_random_affine.default <- function(img, degrees, translate=NULL, scale=NULL,
                                            shear=NULL, resample=0, fillcolor=0) {

  if (length(degrees) == 1) {

    if (degrees < 0)
      value_error("degrees must be positive if it's a single value")

    degrees <- c(-degrees, degrees)

  } else if (length(degrees) != 2) {
    value_error("degrees must be length 1 or 2")
  }


  if (!is.null(translate)) {

    if (length(translate) != 2)
      value_error("translate must be length 2")

    if (any(translate > 1) || any(translate < 0))
      value_error("translate must be between 0 and 1")

  }

  if (!is.null(scale)) {

    if (length(scale) != 2)
      value_error("scale must be length 2")

    if (any(scale < 0))
      value_error("scale must be positive")

  }

  if (!is.null(shear)) {

    if (length(shear) == 1) {

      if (shear < 0)
        value_error("shear must be positive if it's a single value")

      degrees <- c(-degrees, degrees)

    } else if (!length(shear) %in% c(2, 4)) {
      value_error("shear's length must be 1, 2, or 4")
    }

  }


  img_size <- get_image_size(img)

  ret <- get_random_affine_params(degrees, translate, scale, shear, img_size)

  transform_affine(img, ret[[1]], ret[[2]], ret[[3]], ret[[4]],
                   resample=resample, fillcolor=fillcolor)
}

#' @export
transform_grayscale.default <- function(img, num_output_channels) {
  not_implemented_for_class(img)
}

#' @export
transform_random_grayscale.default <- function(img, p = 0.1) {
  not_implemented_for_class(img)
}

get_random_perspective_params <- function(width, height, distortion_scale) {
  half_height <- height %/% 2
  half_width <- width %/% 2
  topleft <- c(
    as.integer(torch::torch_randint(1 + 0,as.integer(distortion_scale * half_width) + 1, size=1)),
    as.integer(torch::torch_randint(1 + 0,as.integer(distortion_scale * half_height) + 1, size=1))
  )
  topright <- c(
    as.integer(torch::torch_randint(1 + width -as.integer(distortion_scale * half_width) - 1, width, size=1)),
    as.integer(torch::torch_randint(1 + 0,as.integer(distortion_scale * half_height) + 1, size=1))
  )
  botright <- c(
    as.integer(torch::torch_randint(1 + width -as.integer(distortion_scale * half_width) - 1, width, size=1)),
    as.integer(torch::torch_randint(1 + height -as.integer(distortion_scale * half_height) - 1, height, size=1))
  )
  botleft <- c(
    as.integer(torch::torch_randint(1 + 0,as.integer(distortion_scale * half_width) + 1, size=1)),
    as.integer(torch::torch_randint(1 + height -as.integer(distortion_scale * half_height) - 1, height, size=1))
  )
  startpoints <- list(c(1 + 0, 1 + 0), c(1 + width - 1, 1 + 0), c(1 + width - 1, 1 + height - 1), c(1 + 0, 1 + height - 1))
  endpoints <- list(topleft, topright, botright, botleft)
  list(startpoints, endpoints)
}

#' @export
transform_random_perspective.default <- function(img, distortion_scale=0.5, p=0.5,
                                                 interpolation=2, fill=0) {
  if (stats::runif(1) < p) {
    img_size <- get_image_size(img)
    params <- get_random_perspective_params(img_size[1], img_size[2], distortion_scale)
    img <- transform_perspective(img, params[[1]], params[[2]], interpolation, fill)
  }

  img
}

#' @export
transform_random_erasing.default <- function(img, p=0.5, scale=c(0.02, 0.33),
                                             ratio=c(0.3, 3.3), value=0, inplace=FALSE) {
  not_implemented_for_class(img)
}

# Other methods -----------------------------------------------------------

#' @export
transform_crop.default <- function(img, top, left, height, width) {
  not_implemented_for_class(img)
}

#' @export
transform_hflip.default <- function(img) {
  not_implemented_for_class(img)
}

#' @export
transform_vflip.default <- function(img) {
  not_implemented_for_class(img)
}

#' @export
transform_resized_crop.default <- function(img, top, left, height, width, size,
                                   interpolation = 2) {
  img <- transform_crop(img, top, left, height, width)
  img <- transform_resize(img, size, interpolation)
  img
}

#' @export
transform_adjust_brightness.default <- function(img, brightness_factor) {
  not_implemented_for_class(img)
}

#' @export
transform_adjust_contrast.default <- function(img, contrast_factor) {
  not_implemented_for_class(img)
}

#' @export
transform_adjust_hue.default <- function(img, hue_factor) {
  not_implemented_for_class(img)
}

#' @export
transform_adjust_saturation.default <- function(img, saturation_factor) {
  not_implemented_for_class(img)
}

#' @export
transform_rotate.default <- function(img, angle, resample = 0, expand = FALSE,
                                     center = NULL, fill = NULL) {
  not_implemented_for_class(img)
}

#' @export
transform_affine.default <- function(img, angle, translate, scale, shear,
                                     resample = 0, fillcolor = NULL) {
  not_implemented_for_class(img)
}

#' @export
transform_perspective.default <- function(img, startpoints, endpoints, interpolation = 2,
                                  fill = NULL) {
  not_implemented_for_class(img)
}

#' @export
transform_adjust_gamma.default <- function(img, gamma, gain = 1) {
  not_implemented_for_class(img)
}


# Helpers -----------------------------------------------------------------

not_implemented_for_class <- function(x) {
  not_implemented_error("not implemented for {class(x)}")
}

