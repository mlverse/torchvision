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
#' url <- "https://upload.wikimedia.org/wikipedia/commons/9/9a/Aeroflot_Airbus_A330_Kustov.jpg"
#'
#' img <- base_loader(url) |>
#'   transform_to_tensor() |>
#'   transform_resize(c(300, 500))
#'
#' item <- list(
#'   x = img,
#'   y = list(
#'     boxes = torch_tensor(matrix(c(40, 95, 475, 180), ncol = 4), dtype = torch_float32()),
#'     labels = "airplane",
#'     image_height = 300L,
#'     image_width = 500L
#'   )
#' )
#' class(item) <- c("image_with_bounding_box", "list")
#'
#' item_rot <- item_transform_rotate(item, angle = 355)
#' item_rot$y$boxes  # (N, 5) tensor in xyxyr format
#'
#' before <- draw_bounding_boxes(item, colors = "blue", width = 4)
#' after <- draw_bounding_boxes(item_rot, colors = "red", width = 4)
#' tensor_image_browse(after)
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
