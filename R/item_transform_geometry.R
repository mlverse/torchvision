#' Convert bounding boxes to rotated format
#'
#' Converts bounding boxes of a detection item from
#' \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} (xyxy) format to
#' \eqn{(x_{min}, y_{min}, x_{max}, y_{max}, r)} (xyxyr) format, where
#' \eqn{r} is the rotation angle in degrees (counter-clockwise). The image
#' is left unchanged. For axis-aligned boxes, \eqn{r = 0}.
#'
#' @param x An object of class \code{image_with_bounding_box} or a dataset that
#'   returns \code{image_with_bounding_box} items via \code{.getitem()}.
#' @param angle (numeric): Rotation angle in degrees (counter-clockwise).
#'   Default is \code{0}.
#'
#' @return An object of class \code{image_with_rotated_box} with the same
#'   structure as the input, except that
#'   \code{$y$boxes} is a tensor of shape \code{(N, 5)} in xyxyr format.
#'   When applied to a dataset, returns the same dataset with its
#'   \code{.getitem} method modified to return \code{image_with_rotated_box}
#'   items.
#'
#' @examples
#' \dontrun{
#' # Convert a single item with a rotation angle
#' ds <- coco_detection_dataset(train = FALSE, year = "2017", download = TRUE)
#' item <- ds[1]
#' rotated_item <- item_transform_rotate(item, angle = 30)
#' rotated_item$y$boxes  # (N, 5) tensor in xyxyr format
#'
#' # Wrap a dataset
#' ds_rotated <- item_transform_rotate(ds, angle = 30)
#' rotated_item <- ds_rotated[1]
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
item_transform_rotate <- function(x, angle = 0) {
  UseMethod("item_transform_rotate", x)
}

#' @export
item_transform_rotate.image_with_bounding_box <- function(x, angle = 0) {
  x$y$boxes <- box_xyxy_to_xyxyr(x$y$boxes, angle = angle)
  class(x) <- c("image_with_rotated_box", setdiff(class(x), "image_with_bounding_box"))
  x
}

#' @export
item_transform_rotate.dataset <- function(x, angle = 0) {
  original_getitem <- x$.getitem
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_rotate(item, angle = angle)
  }
  x
}
