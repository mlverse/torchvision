#' @export
`transform_to_tensor.magick-image` <- function(img) {
  img <- as.integer(magick::image_data(img, channels = "rgb"))
  img <- torch::torch_tensor(img)$permute(c(3,1,2))
  img <- img$to(dtype = torch::torch_float32())
  img <- img$contiguous()
  img <- img$div(255)

  img
}

#' @export
`transform_resize.magick-image` <- function(img, size, interpolation = 2) {

  interpolation_modes <- c(
    "0" = "Pint", # nearest,
    "2" = "Triangle", # bilinear
    "3" = "Catrom" # bicubic
  )

  if (is.numeric(interpolation))
    interpolation <- interpolation_modes[names(interpolation_modes) == interpolation]

  if (length(size) == 1) {

    w <- magick::image_info(img)$width
    h <- magick::image_info(img)$height

    if (w < h)
      size <- paste0(size, "x")
    else
      size <- paste0("x", size)

  } else {
    size <- paste0(paste0(size, collapse = "x"), "!")
  }

  magick::image_resize(img, geometry = size, filter = interpolation)
}

#' @export
`transform_crop.magick-image` <- function(img, top, left, height, width) {
  magick::image_crop(
    img,
    paste0(height, "x", width, "+", left, "+", top)
  )
}

#' @export
`transform_hflip.magick-image` <- function(img) {
  magick::image_flip(img)
}

#' @export
`transform_sahi_crop.magick-image` <- function(x, size = c(512L, 512L), overlap_size_ratio = c(0.2, 0.2)) {

  img_info <- magick::image_info(x)
  image_height <- img_info$height
  image_width <- img_info$width

  crop_height <- as.integer(size[1])
  crop_width <- as.integer(size[2])

  overlap_h <- round(crop_height * overlap_size_ratio[1])
  overlap_w <- round(crop_width * overlap_size_ratio[2])

  step_h <- max(crop_height - overlap_h, 1L)
  step_w <- max(crop_width - overlap_w, 1L)

  if (crop_height >= image_height && crop_width >= image_width) {
    return(list(
      images = list(x),
      crop_windows = list(list(top = 0, left = 0, height = image_height, width = image_width))
    ))
  }

  n_h <- max(ceiling((image_height - crop_height) / step_h) + 1, 1L)
  n_w <- max(ceiling((image_width - crop_width) / step_w) + 1, 1L)

  images <- list()
  crop_windows <- list()

  for (i in seq_len(n_h)) {
    top <- (i - 1L) * step_h
    if (top + crop_height > image_height)
      top <- image_height - crop_height
    for (j in seq_len(n_w)) {
      left <- (j - 1L) * step_w
      if (left + crop_width > image_width)
        left <- image_width - crop_width

      crop <- transform_crop(x, top, left, crop_height, crop_width)
      images <- c(images, list(crop))
      crop_windows <- c(crop_windows, list(list(
        top = as.integer(top),
        left = as.integer(left),
        height = as.integer(crop_height),
        width = as.integer(crop_width)
      )))
    }
  }

  list(images = images, crop_windows = crop_windows)
}

# Utils -------------------------------------------------------------------

#' @method get_image_size magick-image
#' @export
`get_image_size.magick-image` <- function(img) {
  info <- magick::image_info(img)
  c(info$width, info$height)
}

