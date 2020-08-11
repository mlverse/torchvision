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
transform_lambda.default <- function(img) {
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

}
#' @export
transform_random_resized_crop.default <- function(img) {

}
#' @export
transform_random_sized_crop.default <- function(img) {

}
#' @export
transform_five_crop.default <- function(img) {

}
#' @export
transform_ten_crop.default <- function(img) {

}
#' @export
transform_linear_transformation.default <- function(img) {

}
#' @export
transform_color_jitter.default <- function(img) {

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

# Helpers -----------------------------------------------------------------

not_implemented_for_class <- function(x) {
  not_implemented_error("not implemented for ", class(x))
}

