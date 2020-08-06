
#' Resize the input Magick Image to the given size.
#'
#' @param img (Magick Image): Image to be resized.
#' @param size (sequence or int): Desired output size. If size is a sequence like
#'   (h, w), the output size will be matched to this. If size is an int,
#'   the smaller edge of the image will be matched to this number maintaining
#'   the aspect ratio. i.e, if height > width, then image will be rescaled to
#'   \eqn{\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)}.
#'   For compatibility reasons with `tft_resize`, if a tuple or list
#'   of length 1 is provided, it is interpreted as a single int.
#' @param interpolation (int, optional): Desired interpolation. An integer `0 = nearest`,
#'   `2 = bilinear` and `3 = bicubic` or a name from [magick::filter_types()].
#'
#' @return Magick Image: Resized image.
#'
#' @export
tfm_resize <- function(img, size, interpolation = 2) {

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
