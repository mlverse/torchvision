#' Rotate both image and bounding boxes by an angle
#'
#' Rotates the image and bounding boxes of a detection item by a given angle
#' in degrees (counter-clockwise). The image is rotated using the same
#' mechanism as [transform_rotate()], and the bounding boxes are converted to
#' \eqn{(x_{min}, y_{min}, x_{max}, y_{max}, r)} (xyxyr) format via
#' [box_xyxy_to_xyxyr()], where \eqn{r} is the rotation angle in degrees.
#'
#' @param x An object of class \code{image_with_bounding_box} or a dataset that
#'   returns \code{image_with_bounding_box} items via \code{.getitem()}.
#' @param angle (numeric): Rotation angle in degrees (counter-clockwise).
#'   Default is \code{0}.
#' @param expand (logical): If \code{TRUE}, expands the output image canvas to
#'   fit the entire rotated image, and translates bounding boxes accordingly.
#'   If \code{FALSE} (default), the output has the same size as the input and
#'   corners may be clipped.
#' @param interpolation (integer, optional): Interpolation mode. \code{0} for
#'   nearest, \code{2} for bilinear. Default is \code{0} (nearest).
#' @param fill (numeric vector or numeric): Pixel fill value for area outside
#'   the rotated image. Default is \code{0}.
#'
#' @return An object of class \code{image_with_rotated_box} with the same
#'   structure as the input, except that:
#'   \itemize{
#'     \item \code{$x} is the rotated image tensor
#'     \item \code{$y$boxes} is a tensor of shape \code{(N, 5)} in xyxyr format
#'   }
#'   When applied to a dataset, returns the same dataset with its
#'   \code{.getitem} method modified to return \code{image_with_rotated_box}
#'   items.
#'
#' @examples
#' \dontrun{
#' # Rotate a single item by 30 degrees
#' ds <- coco_detection_dataset(train = FALSE, year = "2017", download = TRUE)
#' item <- ds[1]
#' rotated_item <- item_transform_rotate(item, angle = 30)
#' rotated_item$x  # rotated image tensor
#' rotated_item$y$boxes  # (N, 5) tensor in xyxyr format
#'
#' # Rotate with canvas expansion
#' rotated_item <- item_transform_rotate(item, angle = 45, expand = TRUE)
#'
#' # Wrap a dataset
#' ds_rotated <- item_transform_rotate(ds, angle = 30)
#' rotated_item <- ds_rotated[1]
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
item_transform_rotate <- function(x, angle = 0, expand = FALSE,
                                  interpolation = 0, fill = 0) {
  UseMethod("item_transform_rotate", x)
}

#' @export
item_transform_rotate.image_with_bounding_box <- function(x, angle = 0,
                                                          expand = FALSE,
                                                          interpolation = 0,
                                                          fill = 0) {
  # Rotate the image
  x$x <- transform_rotate(x$x, angle = angle, interpolation = interpolation,
                          expand = expand, fill = fill)

  # Compute the canvas shift when expand = TRUE
  shift_x <- 0
  shift_y <- 0
  if (expand) {
    orig_h <- x$y$image_height
    orig_w <- x$y$image_width
    new_h <- x$x$size(-2)
    new_w <- x$x$size(-1)
    shift_x <- (new_w - orig_w) / 2
    shift_y <- (new_h - orig_h) / 2
  }

  # Rotate bounding boxes
  x$y$boxes <- box_xyxy_to_xyxyr(x$y$boxes, angle = angle)

  # Translate boxes when canvas was expanded
  if (expand && (shift_x != 0 || shift_y != 0)) {
    x$y$boxes[, 1] <- x$y$boxes[, 1] + shift_x  # xmin
    x$y$boxes[, 2] <- x$y$boxes[, 2] + shift_y  # ymin
    x$y$boxes[, 3] <- x$y$boxes[, 3] + shift_x  # xmax
    x$y$boxes[, 4] <- x$y$boxes[, 4] + shift_y  # ymax
    x$y$image_height <- new_h
    x$y$image_width <- new_w
  }

  class(x) <- c("image_with_rotated_box", setdiff(class(x), "image_with_bounding_box"))
  x
}

#' @export
item_transform_rotate.dataset <- function(x, angle = 0, expand = FALSE,
                                          interpolation = 0, fill = 0) {
  original_getitem <- x$.getitem
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_rotate(item, angle = angle, expand = expand,
                          interpolation = interpolation, fill = fill)
  }
  x
}
