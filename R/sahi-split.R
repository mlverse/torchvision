#' @rdname transform_sahi_crop
#' @export
prepare_sahi_split <- function(x, size = c(512L, 512L), overlap_size_ratio = c(0.2, 0.2)) {
  UseMethod("prepare_sahi_split", x)
}

#' @export
prepare_sahi_split.numeric <- function(x, size = c(512L, 512L), overlap_size_ratio = c(0.2, 0.2)) {
  if (length(x) != 2)
    value_error("x must be a numeric vector of length 2: c(height, width).")

  image_height <- as.integer(x[1])
  image_width <- as.integer(x[2])

  if (image_height <= 0 || image_width <= 0)
    value_error("Image dimensions must be positive integers.")

  compute_sahi_split(image_height, image_width, size, overlap_size_ratio)
}

#' @export
prepare_sahi_split.torch_tensor <- function(x, size = c(512L, 512L), overlap_size_ratio = c(0.2, 0.2)) {
  check_img(x)
  image_height <- x$size(-2)
  image_width <- x$size(-1)
  compute_sahi_split(image_height, image_width, size, overlap_size_ratio)
}

#' @export
`prepare_sahi_split.magick-image` <- function(x, size = c(512L, 512L), overlap_size_ratio = c(0.2, 0.2)) {
  info <- magick::image_info(x)
  compute_sahi_split(info$height, info$width, size, overlap_size_ratio)
}

#' @export
prepare_sahi_split.dataset <- function(x, size = c(512L, 512L), overlap_size_ratio = c(0.2, 0.2)) {
  meta <- x$image_metadata
  if (!is.null(meta) && is.list(meta)) {
    first <- meta[[1]]
    image_height <- first$height
    image_width <- first$width
    if (!is.null(image_height) && !is.null(image_width))
      return(compute_sahi_split(image_height, image_width, size, overlap_size_ratio))
  }
  item <- x$.getitem(1)
  im <- item$x
  if (inherits(im, "torch_tensor")) {
    image_height <- im$size(-2)
    image_width <- im$size(-1)
  } else if (inherits(im, "array")) {
    image_height <- dim(im)[1]
    image_width <- dim(im)[2]
  } else {
    value_error("Cannot determine image dimensions from dataset.")
  }
  compute_sahi_split(image_height, image_width, size, overlap_size_ratio)
}

# Internal: compute crop windows and return a sahi_split object
compute_sahi_split <- function(image_height, image_width, size, overlap_size_ratio) {

  if (length(size) != 2)
    value_error("Please provide only 2 dimensions (h, w) for size.")

  if (length(overlap_size_ratio) != 2)
    value_error("Please provide only 2 overlap ratios (overlap_h, overlap_w).")

  crop_height <- as.integer(size[1])
  crop_width <- as.integer(size[2])

  overlap_h <- round(crop_height * overlap_size_ratio[1])
  overlap_w <- round(crop_width * overlap_size_ratio[2])

  step_h <- max(as.integer(crop_height - overlap_h), 1L)
  step_w <- max(as.integer(crop_width - overlap_w), 1L)

  if (crop_height >= image_height && crop_width >= image_width) {
    crop_windows <- list(list(
      top = 1L,
      left = 1L,
      height = as.double(image_height),
      width = as.double(image_width)
    ))
  } else {
    n_h <- max(ceiling((image_height - crop_height) / step_h) + 1L, 1L)
    n_w <- max(ceiling((image_width - crop_width) / step_w) + 1L, 1L)

    tops <- seq(0L, by = step_h, length.out = n_h)
    tops <- pmin(tops, image_height - crop_height)
    lefts <- seq(0L, by = step_w, length.out = n_w)
    lefts <- pmin(lefts, image_width - crop_width)

    crop_windows <- vector("list", n_h * n_w)
    idx <- 1L

    for (top in tops)
      for (left in lefts) {
        crop_windows[[idx]] <- list(
          top = as.double(top + 1L),
          left = as.double(left + 1L),
          height = as.double(crop_height),
          width = as.double(crop_width)
        )
        idx <- idx + 1L
      }
  }

  structure(
    list(
      crop_windows = crop_windows,
      size = size,
      overlap_size_ratio = overlap_size_ratio,
      image_height = image_height,
      image_width = image_width
    ),
    class = "sahi_split"
  )
}
