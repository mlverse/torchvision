#' Prepare SAHI crop split parameters
#'
#' Precomputes crop windows for the Slicing Aided Hyper Inference (SAHI)
#' approach. The returned `sahi_split` object can be passed to
#' `transform_sahi_crop()` and `target_transform_sahi_crop()` so that
#' transforms remain type-invariant (returning the same type as their input).
#'
#' @param x Object to determine image dimensions from. Can be:
#'   * A numeric vector of length 2: `c(height, width)`
#'   * A `torch_tensor`: dimensions extracted from the tensor shape
#'   * A `magick-image`: dimensions extracted from image metadata
#'   * A dataset (e.g. `coco_detection_dataset`): dimensions extracted from
#'     the first sample's metadata or loaded image
#' @param size Integer vector of length 2 containing crop height and width
#'   in the form `c(height, width)`.
#' @param overlap_size_ratio Numeric vector of length 2 containing vertical
#'   and horizontal overlap ratios in the form
#'   `c(overlap_height_ratio, overlap_width_ratio)`.
#'
#' @return An object of class `sahi_split` containing:
#'   \describe{
#'     \item{crop_windows}{List of crop windows with 
#'      each window is a list with `top`, `left`, `height` and `width`
#'       fields specifying the region to crop from the original image.}
#'     \item{size}{Integer vector of length 2: the crop `c(height, width)` used
#'       to generate the split.}
#'     \item{overlap_size_ratio}{Numeric vector of length 2: the vertical and
#'       horizontal overlap ratios `c(overlap_h, overlap_w)` used.}
#'     \item{image_height}{Height of the original image (integer).}
#'     \item{image_width}{Width of the original image (integer).}
#'   }
#'
#' @examples
#' \dontrun{
#' # From a torch tensor
#' img_url <- "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg"
#' img <- base_loader(img_url) %>% transform_to_tensor()
#' sp <- prepare_sahi_split(img, size = c(512, 512), overlap_size_ratio = c(0.2, 0.2))
#'
#' crops <- transform_sahi_crop(img, sp)
#' grid <- vision_make_grid(crops, scale = TRUE, num_rows = 3)
#' tensor_image_browse(grid)
#'
#' # From a dataset (SAHI as part of the transform pipeline)
#' ds <- coco_detection_dataset(train = FALSE, year = "2017", download = TRUE)
#' sp_ds <- prepare_sahi_split(ds, size = c(200, 200), overlap_size_ratio = c(0.2, 0.2))
#'
#' ds <- coco_detection_dataset(train = FALSE, year = "2017",
#'   transform = . %>% transform_to_tensor() %>%
#'     transform_sahi_crop(sp_ds))
#'
#' item <- ds[1]
#' grid <- vision_make_grid(item$x, scale = TRUE, num_rows = 3)
#' tensor_image_browse(grid)
#' }
#'
#' @family sahi
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
  if (inherits(x, "R6")) {
    # Attempt to read image_metadata (used by COCO-style datasets)
    meta <- x$image_metadata
    if (!is.null(meta) && is.list(meta)) {
      first <- meta[[1]]
      image_height <- first$height
      image_width <- first$width
      if (!is.null(image_height) && !is.null(image_width))
        return(compute_sahi_split(image_height, image_width, size, overlap_size_ratio))
    }
    # Fall back: load the first sample to determine dimensions
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
  } else {
    not_implemented_for_class(x)
  }
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
