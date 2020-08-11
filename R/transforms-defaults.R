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
transform_resize.default <- function(img, size) {
  not_implemented_for_class(img)
}

#' @export
transform_center_crop.default <- function(img, size) {

  output_size <- size

  if (length(size) == 1)
    output_size <- rep(size, 2)

  output_size <- as.integer(output_size)

  size <- get_image_size(img)

  image_width <- size[1]
  image_height <- size[2]

  crop_height <- output_size[1]
  crop_width <- output_size[2]

  crop_top <- as.integer((image_height - crop_height) / 2)
  crop_left <- as.integer((image_width - crop_width) / 2)

  transform_crop(img, crop_top, crop_left, crop_height, crop_width)
}

#' @export
transform_pad.default <- function(img, padding, fill = 0, padding_mode = "constant") {
  not_implemented_for_class(img)
}

#' @export
transform_random_apply.default <- function(img, transforms, p = 0.5) {

  if (p < runif(1))
    return(img)

  for (tf in transforms) {
    img <- tf(img)
  }

  img
}

#' @export
transform_random_choice.default <- function(img, transforms) {
  i <- sample.int(length(transforms))
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

  params <- get_random_crop_params(img, self.size)

  transform_crop(img, params[1], params[2], params[3], params[4])
}

#' @export
transform_random_horizontal_flip.default <- function(img, p = 0.5) {

  if (runif(1) < p)
    img <- transform_hflip(img)

  img
}


#' @export
transform_random_vertical_flip.default <- function(img) {

  if (runif(1) < p)
    img <- transform_vflip(img)

  img

}

get_random_resized_crop_params <- function(img, scale, ratio) {

  image_size <- get_image_size(img)
  width <- img_size[1]; height <- img_size[2]

  area <- height * width

  for (i in 1:10) {
    target_area <- as.numeric(area * torch::torch_empty(1)$uniform_(scale[1], scale[2]))
    log_ratio <- torch::torch_log(torch::torch_tensor(ratio))
    aspect_ratio <-  as.numeric(torch::torch_exp(
      torch::torch_empty(1)$uniform_(log_ratio[1], log_ratio[2])
    ))

    w <- as.integer(round(sqrt(target_area * aspect_ratio)))
    h <- as.integer(round(sqrt(target_area / aspect_ratio)))


    if (0 < w && w <= width && 0 < h && h <= height) {

      i = as.integer(torch::torch_randint(1, height - h + 1, size=1))
      j = as.integer(torch::torch_randint(0, width - w + 1, size=1))

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
    i <- (height - h) %/% 2
    j <- (width - w) %/% 2
  }

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
transform_five_crop.default <- function(img) {
  not_implemented_for_class(img)
}

#' @export
transform_ten_crop.default <- function(img) {
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
    brightness_factor <- runif(1, min = brightness[1], max = brightness[2])
    transforms <- append(
      transforms,
      list(function(img) transform_adjust_brightness(img, brightness_factor))
    )
  }

  if (!is.null(contrast)) {
    contrast_factor <- runif(1, contrast[1], contrast[2])
    transforms <- append(
      transforms,
      list(function(img) transform_adjust_contrast(img, contrast_factor))
    )
  }

  if (!is.null(saturation)) {
    saturation_factor <- runif(1, saturation[1], saturation[2])
    transforms <- append(
      transforms,
      list(function(img) transform_adjust_saturation(img, saturation_factor))
    )
  }

  if (!is.null(hue)) {
    hue_factor <- runif(1, hue[1], hue[2])
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

  } else if (length(value == 2)) {

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

#' @export
transform_random_rotation.default <- function(img) {

}

#' @export
transform_random_affine.default <- function(img) {

}

#' @export
transform_grayscale.default <- function(img) {

}

#' @export
transform_random_grayscale.default <- function(img) {

}

#' @export
transform_random_perspective.default <- function(img) {

}

#' @export
transform_random_erasing.default <- function(img) {

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

# Helpers -----------------------------------------------------------------

not_implemented_for_class <- function(x) {
  not_implemented_error("not implemented for ", class(x))
}

