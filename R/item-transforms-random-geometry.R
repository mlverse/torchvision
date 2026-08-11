#' Randomly horizontally flip a dataset item
#'
#' Randomly flips the image inside a dataset item horizontally with a given
#' probability. When a flip occurs, the same coordinate adjustments used by
#' \code{\link{item_transform_hflip}} are applied to bounding boxes and masks.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @param p Probability of the item being flipped. Default is 0.5.
#'
#' @return A dataset item of the same class. With probability \code{p}, the
#'   image and targets are horizontally flipped; otherwise they are returned
#'   unchanged.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = {"CAT"}))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_random_horizontal_flip(before)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot)), scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_random_transforms
#'
#' @export
item_transform_random_horizontal_flip <- function(x, p = 0.5) {
  UseMethod("item_transform_random_horizontal_flip", x)
}

#' @export
item_transform_random_horizontal_flip.default <- function(x, p = 0.5) {
  cli_abort(
    "{.fn item_transform_random_horizontal_flip} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To flip a raw image tensor, use {.fn transform_random_horizontal_flip} instead."
  )
}

#' @export
item_transform_random_horizontal_flip.dataset <- function(x, p = 0.5) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_random_horizontal_flip(item, p = p)
  }
  x
}

#' @export
item_transform_random_horizontal_flip.image_with_bounding_box <- function(x, p = 0.5) {
  if (stats::runif(1) < p) {
    x <- item_transform_hflip(x)
  }
  x
}

#' @export
item_transform_random_horizontal_flip.image_with_segmentation_mask <- function(x, p = 0.5) {
  if (stats::runif(1) < p) {
    x <- item_transform_hflip(x)
  }
  x
}

#' @export
item_transform_random_horizontal_flip.image_with_rotated_box <- function(x, p = 0.5) {
  if (stats::runif(1) < p) {
    x <- item_transform_hflip(x)
  }
  x
}

#' Randomly vertically flip a dataset item
#'
#' Randomly flips the image inside a dataset item vertically with a given
#' probability. When a flip occurs, the same coordinate adjustments used by
#' \code{\link{item_transform_vflip}} are applied to bounding boxes and masks.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @param p Probability of the item being flipped. Default is 0.5.
#'
#' @return A dataset item of the same class. With probability \code{p}, the
#'   image and targets are vertically flipped; otherwise they are returned
#'   unchanged.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = {"CAT"}))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_random_vertical_flip(before)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot)), num_rows = 1, scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_random_transforms
#'
#' @export
item_transform_random_vertical_flip <- function(x, p = 0.5) {
  UseMethod("item_transform_random_vertical_flip", x)
}

#' @export
item_transform_random_vertical_flip.default <- function(x, p = 0.5) {
  cli_abort(
    "{.fn item_transform_random_vertical_flip} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To flip a raw image tensor, use {.fn transform_random_vertical_flip} instead."
  )
}

#' @export
item_transform_random_vertical_flip.dataset <- function(x, p = 0.5) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_random_vertical_flip(item, p = p)
  }
  x
}

#' @export
item_transform_random_vertical_flip.image_with_bounding_box <- function(x, p = 0.5) {
  if (stats::runif(1) < p) {
    x <- item_transform_vflip(x)
  }
  x
}

#' @export
item_transform_random_vertical_flip.image_with_segmentation_mask <- function(x, p = 0.5) {
  if (stats::runif(1) < p) {
    x <- item_transform_vflip(x)
  }
  x
}

#' @export
item_transform_random_vertical_flip.image_with_rotated_box <- function(x, p = 0.5) {
  if (stats::runif(1) < p) {
    x <- item_transform_vflip(x)
  }
  x
}

#' Randomly crop a dataset item
#'
#' Crops the image inside a dataset item at a random location. When a crop
#' occurs, the same coordinate adjustments used by
#' \code{\link{item_transform_crop}} are applied to bounding boxes and masks so
#' that the targets stay aligned with the cropped image.
#'
#' If the image is smaller than the requested crop size, it can optionally be
#' padded first (see \code{padding} and \code{pad_if_needed}).
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @param size Desired output size of the crop. If \code{size} is an
#'   int instead of a sequence like \code{c(h, w)}, a square crop
#'   \code{c(size, size)} is made. If a sequence of length 1 is provided it is
#'   interpreted as \code{c(size[1], size[1])}.
#' @param padding (int or vector, optional) Optional padding on each border
#'   of the image, applied before cropping. Default is \code{NULL}. If a single
#'   int is provided this is used to pad all borders. If a vector of length 2 is
#'   provided this is the padding on left/right and top/bottom respectively. If
#'   a vector of length 4 is provided this is the padding for the left, top,
#'   right and bottom borders respectively.
#' @param pad_if_needed (logical) It will pad the image if smaller than the
#'   desired size to avoid raising an exception. Since cropping is done
#'   after padding, the padding seems to be done at a random offset.
#' @param fill (number or tuple) Pixel fill value for constant fill. Default is
#'   0. If a tuple of length 3, it is used to fill R, G, B channels
#'   respectively. This value is only used when the padding_mode is constant.
#' @param padding_mode (str) Type of padding. Should be: constant, edge,
#'   reflect or symmetric. Default is constant.
#'
#' @return A dataset item of the same class with the image and target randomly
#'   cropped to the requested size.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = {"CAT"}))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_random_crop(before, size = c(800, 1200))
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' # the crop changes the image size, so resize before stacking into the grid
#' grid <- vision_make_grid(
#'   torch_stack(list(transform_resize(before_plot, c(600, 600)),
#'                    transform_resize(after_plot, c(600, 600)))),
#'   scale = TRUE
#' )
#' tensor_image_browse(grid)
#' }
#'
#' @family item_random_transforms
#'
#' @export
item_transform_random_crop <- function(x, size, padding = NULL, pad_if_needed = FALSE,
                                       fill = 0, padding_mode = "constant") {
  UseMethod("item_transform_random_crop", x)
}

#' @export
item_transform_random_crop.default <- function(x, size, padding = NULL, pad_if_needed = FALSE,
                                               fill = 0, padding_mode = "constant") {
  cli_abort(
    "{.fn item_transform_random_crop} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To randomly crop a raw image tensor, use {.fn transform_random_crop} instead."
  )
}

#' @export
item_transform_random_crop.dataset <- function(x, size, padding = NULL, pad_if_needed = FALSE,
                                               fill = 0, padding_mode = "constant") {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_random_crop(item, size = size, padding = padding,
                               pad_if_needed = pad_if_needed,
                               fill = fill, padding_mode = padding_mode)
  }
  x
}

#' @export
item_transform_random_crop.image_with_bounding_box <- function(x, size, padding = NULL,
                                                              pad_if_needed = FALSE,
                                                              fill = 0,
                                                              padding_mode = "constant") {
  output_size <- as.integer(if (length(size) == 1) rep(size, 2) else size)

  if (!is.null(padding)) {
    x <- item_transform_pad(x, padding, fill = fill, padding_mode = padding_mode)
  }

  if (pad_if_needed) {
    img_size <- get_image_size(x$x)
    if (img_size[1] < output_size[2]) {
      x <- item_transform_pad(x, c(output_size[2] - img_size[1], 0),
                              fill = fill, padding_mode = padding_mode)
    }
    img_size <- get_image_size(x$x)
    if (img_size[2] < output_size[1]) {
      x <- item_transform_pad(x, c(0, output_size[1] - img_size[2]),
                              fill = fill, padding_mode = padding_mode)
    }
  }

  img_size <- get_image_size(x$x)
  if (img_size[1] < output_size[2] || img_size[2] < output_size[1]) {
    cli_abort(
      "Required crop size ({output_size[1]}, {output_size[2]}) is larger than input image size ({img_size[2]}, {img_size[1]})."
    )
  }

  params <- get_random_crop_params(x$x, output_size)

  item_transform_crop(x, top = params[1], left = params[2],
                      height = params[3], width = params[4])
}

#' @export
item_transform_random_crop.image_with_segmentation_mask <- function(x, size, padding = NULL,
                                                                   pad_if_needed = FALSE,
                                                                   fill = 0,
                                                                   padding_mode = "constant") {
  output_size <- as.integer(if (length(size) == 1) rep(size, 2) else size)

  if (!is.null(padding)) {
    x <- item_transform_pad(x, padding, fill = fill, padding_mode = padding_mode)
  }

  if (pad_if_needed) {
    img_size <- get_image_size(x$x)
    if (img_size[1] < output_size[2]) {
      x <- item_transform_pad(x, c(output_size[2] - img_size[1], 0),
                              fill = fill, padding_mode = padding_mode)
    }
    img_size <- get_image_size(x$x)
    if (img_size[2] < output_size[1]) {
      x <- item_transform_pad(x, c(0, output_size[1] - img_size[2]),
                              fill = fill, padding_mode = padding_mode)
    }
  }

  img_size <- get_image_size(x$x)
  if (img_size[1] < output_size[2] || img_size[2] < output_size[1]) {
    cli_abort(
      "Required crop size ({output_size[1]}, {output_size[2]}) is larger than input image size ({img_size[2]}, {img_size[1]})."
    )
  }

  params <- get_random_crop_params(x$x, output_size)

  item_transform_crop(x, top = params[1], left = params[2],
                      height = params[3], width = params[4])
}

#' @export
item_transform_random_crop.image_with_rotated_box <- function(x, size, padding = NULL,
                                                              pad_if_needed = FALSE,
                                                              fill = 0,
                                                              padding_mode = "constant") {
  output_size <- as.integer(if (length(size) == 1) rep(size, 2) else size)

  if (!is.null(padding)) {
    x <- item_transform_pad(x, padding, fill = fill, padding_mode = padding_mode)
  }

  if (pad_if_needed) {
    img_size <- get_image_size(x$x)
    if (img_size[1] < output_size[2]) {
      x <- item_transform_pad(x, c(output_size[2] - img_size[1], 0),
                              fill = fill, padding_mode = padding_mode)
    }
    img_size <- get_image_size(x$x)
    if (img_size[2] < output_size[1]) {
      x <- item_transform_pad(x, c(0, output_size[1] - img_size[2]),
                              fill = fill, padding_mode = padding_mode)
    }
  }

  img_size <- get_image_size(x$x)
  if (img_size[1] < output_size[2] || img_size[2] < output_size[1]) {
    cli_abort(
      "Required crop size ({output_size[1]}, {output_size[2]}) is larger than input image size ({img_size[2]}, {img_size[1]})."
    )
  }

  params <- get_random_crop_params(x$x, output_size)

  item_transform_crop(x, top = params[1], left = params[2],
                      height = params[3], width = params[4])
}
