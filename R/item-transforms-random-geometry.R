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
  random_affine_item(x, degrees, translate, scale, shear, interpolation, fill, center)
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
