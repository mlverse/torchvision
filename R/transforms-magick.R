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
`transform_sahi_crop.magick-image` <- function(x, sahi_split) {

  crops <- lapply(sahi_split$crop_windows, function(cw) {
    # Convert 1-based coords to 0-based for magick
    transform_crop(x, cw$top - 1, cw$left - 1, cw$height, cw$width)
  })

  magick::image_join(crops)
}

# Utils -------------------------------------------------------------------

#' @method get_image_size magick-image
#' @export
`get_image_size.magick-image` <- function(img) {
  info <- magick::image_info(img)
  c(info$width, info$height)
}

