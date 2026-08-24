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
#' @importFrom utils head
#' @importFrom stats runif
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
  if (runif(1) < p) {
    x <- item_transform_hflip(x)
  }
  x
}

#' @export
item_transform_random_horizontal_flip.image_with_segmentation_mask <- function(x, p = 0.5) {
  if (runif(1) < p) {
    x <- item_transform_hflip(x)
  }
  x
}

#' @export
item_transform_random_horizontal_flip.image_with_rotated_box <- function(x, p = 0.5) {
  if (runif(1) < p) {
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

#' Randomly crop a dataset item and resize it
#'
#' Crops the image inside a dataset item to a random area and aspect ratio, then
#' resizes that crop to the given size. For detection items, bounding boxes are
#' cropped along with the image and their coordinates are rescaled to the output
#' size: boxes falling outside the crop are dropped and boxes straddling its
#' border are clipped to it. For segmentation items, the masks are cropped and
#' resized alongside the image with nearest-neighbour sampling, so that they keep
#' their discrete values.
#'
#' The crop is drawn again for every item, so that a dataset wrapped with this
#' transform yields a different crop on each access.
#'
#' The \code{area} field of a detection target, when present, is rescaled by the
#' area ratio between the crop and the output. For rotated boxes, a crop whose
#' aspect ratio differs from the output one maps the box to a parallelogram: the
#' enclosing box stays exact and the angle is that of the transformed box axis,
#' which is exact whenever both edges are scaled alike.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @inheritParams transform_random_resized_crop
#'
#' @return A dataset item of the same class with the image and target cropped
#'   and resized.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = "cat"))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_random_resize_crop(before, size = c(600, 800))
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)
#' tensor_image_browse(before_plot)
#' tensor_image_browse(after_plot)
#' }
#'
#' @family item_random_transforms
#'
#' @importFrom torch torch_atan2 torch_rand
#' @export
item_transform_random_resize_crop <- function(x, size, scale = c(0.08, 1),
                                              ratio = c(3 / 4, 4 / 3),
                                              interpolation = 2) {
  UseMethod("item_transform_random_resize_crop", x)
}

#' @export
item_transform_random_resize_crop.dataset <- function(x, size, scale = c(0.08, 1),
                                                      ratio = c(3 / 4, 4 / 3),
                                                      interpolation = 2) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_random_resize_crop(item, size = size, scale = scale,
                                      ratio = ratio, interpolation = interpolation)
  }
  x
}

#' @export
item_transform_random_resize_crop.default <- function(x, size, scale = c(0.08, 1),
                                                      ratio = c(3 / 4, 4 / 3),
                                                      interpolation = 2) {
  cli_abort(
    "{.fn item_transform_random_resize_crop} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To crop and resize a raw image tensor, use {.fn transform_random_resized_crop} instead."
  )
}

#' @export
item_transform_random_resize_crop.image_with_bounding_box <- function(x, size, scale = c(0.08, 1),
                                                                      ratio = c(3 / 4, 4 / 3),
                                                                      interpolation = 2) {
  params <- get_random_resized_crop_params(x$x, scale, ratio)
  x <- item_transform_crop(x, top = params[1], left = params[2],
                           height = params[3], width = params[4])

  x$x <- transform_resize(x$x, size, interpolation)

  new_spatial <- tail(x$x$shape, 2)
  scale_h <- new_spatial[1] / params[3]
  scale_w <- new_spatial[2] / params[4]

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    boxes[, 1] <- boxes[, 1] * scale_w
    boxes[, 2] <- boxes[, 2] * scale_h
    boxes[, 3] <- boxes[, 3] * scale_w
    boxes[, 4] <- boxes[, 4] * scale_h
    if (boxes$size(2) == 5) {
      boxes[, 5] <- rescale_box_angle(boxes[, 5], scale_w, scale_h)
    }
  }
  x$y$boxes <- boxes
  if (!is.null(x$y$area)) {
    x$y$area <- x$y$area * (scale_w * scale_h)
  }
  x$y$image_height <- new_spatial[1]
  x$y$image_width <- new_spatial[2]

  x
}

#' @export
item_transform_random_resize_crop.image_with_segmentation_mask <- function(x, size, scale = c(0.08, 1),
                                                                           ratio = c(3 / 4, 4 / 3),
                                                                           interpolation = 2) {
  params <- get_random_resized_crop_params(x$x, scale, ratio)
  x <- item_transform_crop(x, top = params[1], left = params[2],
                           height = params[3], width = params[4])

  x$x <- transform_resize(x$x, size, interpolation)

  new_spatial <- tail(x$x$shape, 2)
  masks <- x$y$masks
  x$y$masks <- if (masks$ndim == 2) {
    transform_resize(masks$unsqueeze(1), new_spatial, interpolation = 0)$squeeze(1)
  } else if (masks$size(1) == 0) {
    torch_zeros(c(0, new_spatial), dtype = masks$dtype, device = masks$device)
  } else {
    transform_resize(masks, new_spatial, interpolation = 0)
  }
  x$y$image_height <- new_spatial[1]
  x$y$image_width <- new_spatial[2]

  x
}

#' @export
item_transform_random_resize_crop.image_with_rotated_box <- item_transform_random_resize_crop.image_with_bounding_box

rescale_box_angle <- function(angle_deg, scale_w, scale_h) {
  if (scale_w == scale_h) {
    return(angle_deg)
  }

  rad <- deg2rad(angle_deg)
  rad2deg(torch_atan2(scale_h * torch_sin(rad), scale_w * torch_cos(rad)))
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


#' Randomly apply an affine transformation on a dataset item
#'
#' Draws a random rotation, translation, scale and shear inside the given ranges
#' and applies the resulting affine transformation to the dataset item with
#' \code{\link{item_transform_affine}}, keeping the image size unchanged. Image
#' and target share the same draw, so that they stay aligned.
#'
#' The transformation is drawn again for every item, so that a dataset wrapped
#' with this transform yields a different transformation on each access.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box},
#'   \code{image_with_rotated_box} or \code{image_with_segmentation_mask} object
#'   containing an image tensor and associated target data.
#' @inheritParams transform_random_affine
#' @param fill Fill color for the area outside the transform. Default is
#'   \code{NULL}.
#' @param center (numeric vector of length 2, optional): Optional center of
#'   rotation, \code{c(x, y)}. Default is image center.
#'
#' @return A dataset item with the image and target transformed. Detection items
#'   are returned as \code{image_with_rotated_box}; segmentation items keep their
#'   class.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = "cat",
#'                                  image_height = img$shape[2], image_width = img$shape[3]))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_random_affine(before, degrees = 30, translate = c(0.1, 0.1),
#'                                       scale = c(0.8, 1.2), shear = 10)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)
#' tensor_image_browse(before_plot)
#' tensor_image_browse(after_plot)
#' }
#'
#' @family item_random_transforms
#'
#' @export
item_transform_random_affine <- function(x, degrees, translate = NULL, scale = NULL,
                                         shear = NULL, interpolation = 0, fill = NULL,
                                         center = NULL) {
  UseMethod("item_transform_random_affine", x)
}

#' @export
item_transform_random_affine.default <- function(x, degrees, translate = NULL, scale = NULL,
                                                 shear = NULL, interpolation = 0, fill = NULL,
                                                 center = NULL) {
  cli_abort(
    "{.fn item_transform_random_affine} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To transform a raw image tensor, use {.fn transform_random_affine} instead."
  )
}

#' @export
item_transform_random_affine.dataset <- function(x, degrees, translate = NULL, scale = NULL,
                                                 shear = NULL, interpolation = 0, fill = NULL,
                                                 center = NULL) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_random_affine(item, degrees = degrees, translate = translate,
                                 scale = scale, shear = shear,
                                 interpolation = interpolation, fill = fill,
                                 center = center)
  }
  x
}

#' @export
item_transform_random_affine.image_with_bounding_box <- function(x, degrees, translate = NULL,
                                                                 scale = NULL, shear = NULL,
                                                                 interpolation = 0, fill = NULL,
                                                                 center = NULL) {
  args <- check_random_affine_params(degrees, translate, scale, shear)
  params <- get_random_affine_params(args$degrees, translate, scale, args$shear,
                                     get_image_size(x$x))

  item_transform_affine(x, angle = params[[1]], translate = params[[2]],
                        scale = params[[3]], shear = params[[4]],
                        interpolation = interpolation, fill = fill, center = center)
}

#' @export
item_transform_random_affine.image_with_rotated_box <- function(x, degrees, translate = NULL,
                                                                scale = NULL, shear = NULL,
                                                                interpolation = 0, fill = NULL,
                                                                center = NULL) {
  random_affine_item(x, degrees, translate, scale, shear, interpolation, fill, center)
}

#' @export
item_transform_random_affine.image_with_segmentation_mask <- function(x, degrees, translate = NULL,
                                                                      scale = NULL, shear = NULL,
                                                                      interpolation = 0, fill = NULL,
                                                                      center = NULL) {
  random_affine_item(x, degrees, translate, scale, shear, interpolation, fill, center)
}

random_affine_item <- function(x, degrees, translate, scale, shear, interpolation, fill, center) {
  args <- check_random_affine_params(degrees, translate, scale, shear)
  params <- get_random_affine_params(args$degrees, translate, scale, args$shear,
                                     get_image_size(x$x))

  item_transform_affine(x, angle = params[[1]], translate = params[[2]],
                        scale = params[[3]], shear = params[[4]],
                        interpolation = interpolation, fill = fill, center = center)
}

#' Randomly rotate a dataset item
#'
#' Draws a random angle inside the given range and rotates the dataset item by
#' that angle with \code{\link{item_transform_rotate}}. Image and target share
#' the same draw, so that boxes and masks stay aligned with the rotated image.
#'
#' The angle is drawn again for every item, so that a dataset wrapped with this
#' transform yields a different rotation on each access. Unlike
#' \code{\link{item_transform_rotate}}, the canvas is not expanded by default, so
#' that every item keeps the size it had and a wrapped dataset can still be
#' collated into batches. Pass \code{expand = TRUE} to hold the whole rotated
#' image instead.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box},
#'   \code{image_with_rotated_box} or \code{image_with_segmentation_mask} object
#'   containing an image tensor and associated target data.
#' @param interpolation (integer, optional): Interpolation mode. \code{0} for
#'   nearest, \code{2} for bilinear. Default is \code{2} (bilinear), as in
#'   \code{\link{item_transform_rotate}}. Masks are always resampled with
#'   nearest so that they keep their discrete values.
#' @inheritParams transform_random_rotation
#'
#' @return A dataset item with the image and target rotated. Detection items are
#'   returned as \code{image_with_rotated_box} with boxes in xyxyr format;
#'   segmentation items keep their class.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = "cat"))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_random_rotation(before, degrees = 30)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)
#' tensor_image_browse(before_plot)
#' tensor_image_browse(after_plot)
#' }
#'
#' @family item_random_transforms
#'
#' @export
item_transform_random_rotation <- function(x, degrees, interpolation = 2,
                                           expand = FALSE, fill = 0) {
  UseMethod("item_transform_random_rotation", x)
}

#' @export
item_transform_random_rotation.default <- function(x, degrees, interpolation = 2,
                                                   expand = FALSE, fill = 0) {
  cli_abort(
    "{.fn item_transform_random_rotation} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To rotate a raw image tensor, use {.fn transform_random_rotation} instead."
  )
}

#' @export
item_transform_random_rotation.dataset <- function(x, degrees, interpolation = 2,
                                                   expand = FALSE, fill = 0) {
  force(degrees)
  force(interpolation)
  force(expand)
  force(fill)

  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_random_rotation(item, degrees = degrees,
                                   interpolation = interpolation,
                                   expand = expand, fill = fill)
  }
  x
}

#' @export
item_transform_random_rotation.image_with_bounding_box <- function(x, degrees, interpolation = 2,
                                                                   expand = FALSE, fill = 0) {
  degrees <- check_random_rotation_params(degrees)
  angle <- get_random_rotation_params(degrees)

  item_transform_rotate(x, angle = angle, interpolation = interpolation,
                        expand = expand, fill = fill)
}

#' @export
item_transform_random_rotation.image_with_rotated_box <- item_transform_random_rotation.image_with_bounding_box

#' @export
item_transform_random_rotation.image_with_segmentation_mask <- item_transform_random_rotation.image_with_bounding_box



#' Randomly erase a rectangular region of a dataset item
#'
#' Randomly selects a rectangular region in the image inside a dataset item and
#' erases its pixel values with a given probability. Only the image pixels are
#' modified: bounding boxes and masks are left unchanged.
#'
#' 'Random Erasing Data Augmentation' by Zhong _et al._ See
#' <https://arxiv.org/abs/1708.04896>
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @param p (numeric): Probability that the random erasing operation will be
#'   performed. Default is 0.5.
#' @param scale (numeric vector of length 2): Range of proportion of erased
#'   area against input image.
#' @param ratio (numeric vector of length 2): Range of aspect ratio of erased
#'   area.
#' @param value (numeric vector or numeric or character): Erasing value.
#'   Default is `0`. If a single numeric value, it is used to erase all
#'   pixels. If a numeric vector of length 3, it is used to erase R, G, B
#'   channels respectively. If the string `"random"`, erasing each pixel with
#'   random values.
#' @param inplace (logical): Boolean to make this transform inplace. Default
#'   set to `FALSE`.
#'
#' @return A dataset item of the same class. With probability \code{p}, the
#'   image is randomly erased; otherwise it is returned unchanged.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = "cat"))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_random_erasing(before)
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
item_transform_random_erasing <- function(x, p = 0.5, scale = c(0.02, 0.33), ratio = c(0.3, 3.3),
                                          value = 0, inplace = FALSE) {
  UseMethod("item_transform_random_erasing", x)
}

#' @export
item_transform_random_erasing.default <- function(x, p = 0.5, scale = c(0.02, 0.33), ratio = c(0.3, 3.3),
                                                  value = 0, inplace = FALSE) {
  cli_abort(
    "{.fn item_transform_random_erasing} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To erase a raw image tensor, use {.fn transform_random_erasing} instead."
  )
}

#' @export
item_transform_random_erasing.dataset <- function(x, p = 0.5, scale = c(0.02, 0.33), ratio = c(0.3, 3.3),
                                                  value = 0, inplace = FALSE) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_random_erasing(item, p = p, scale = scale, ratio = ratio,
                                  value = value, inplace = inplace)
  }
  x
}

#' @export
item_transform_random_erasing.image_with_bounding_box <- function(x, p = 0.5, scale = c(0.02, 0.33),
                                                                  ratio = c(0.3, 3.3), value = 0,
                                                                  inplace = FALSE) {
  if (stats::runif(1) < p) {
    img_size <- get_image_size(x$x)
    c(top, left, height, width) %<-% get_random_erasing_params(img_size[2], img_size[1], scale, ratio)
    if (!is.null(top)) {
      # x$x <- erase_tensor_region(x$x, top, left, height, width, value, inplace)
      img_c <- x$x$size(1)

      if (!inplace) {
        x$x <- x$x$clone()
      }

      c(img_w, img_h) %<-% get_image_size(x$x)

      top <- max(0L, top)
      left <- max(0L, left)
      h <- min(height, img_h - top)
      w <- min(width, img_w - left)

      if (!(h <= 0L || w <= 0L)) {
        # inplace region replacement
        region <- x$x$narrow(2, top + 1L, h)$narrow(3, left + 1L, w)

        if (is.character(value)) {
          region$copy_(torch_randn(img_c, h, w, dtype = torch_float32()))
        } else if (length(value) == 1) {
          region$fill_(value)
        } else {
          region$copy_(torch_tensor(value, dtype = x$x$dtype, device = x$x$device)$view(c(img_c, 1, 1)))
        }

      }
    }
  }
  x
}

#' @export
item_transform_random_erasing.image_with_segmentation_mask <- item_transform_random_erasing.image_with_bounding_box

#' @export
item_transform_random_erasing.image_with_rotated_box <- item_transform_random_erasing.image_with_bounding_box

# Sample the location and size of a random erasing rectangle.
#
# Returns a list with 0-indexed `top` and `left` and positive `height` and
# `width`, or NULL when no valid rectangle was found (in which case the image
# is returned unchanged).
get_random_erasing_params <- function(img_h, img_w, scale, ratio) {
  area <- img_h * img_w
  log_ratio <- log(ratio)

  erase_area <- area * stats::runif(10, min = scale[1], max = scale[2])
  aspect_ratio <- exp(stats::runif(10, min = log_ratio[1], max = log_ratio[2]))

  h <- round(sqrt(erase_area * aspect_ratio))
  w <- round(sqrt(erase_area / aspect_ratio))

  valid <- which(h < img_h & w < img_w)
  if (length(valid) == 0L) return(NULL)

  idx <- valid[1L]
  top <- as.integer(floor(stats::runif(1, 0, img_h - h[idx] + 1)))
  left <- as.integer(floor(stats::runif(1, 0, img_w - w[idx] + 1)))

  list(top = top, left = left, height = h[idx], width = w[idx])
}
