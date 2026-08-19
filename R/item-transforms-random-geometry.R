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
#' @importFrom torch torch_atan2
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
