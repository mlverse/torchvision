#' Rotate dataset item
#'
#' Rotates an image by a given angle around its center. The canvas is expanded
#' so that the entire rotated image is visible with no cropping. Empty regions
#' are filled with black.
#'
#' When passed an \code{image_with_bounding_box} object, the bounding boxes are
#' also rotated via an internal call to \code{\link{target_transform_rotate_box}}.
#'
#' @param x A torch tensor of shape \code{(C, H, W)}, or an
#'   \code{image_with_bounding_box} object.
#' @param angle (numeric): Rotation angle in degrees (counter-clockwise).
#'   Default is \code{0}.
#'
#' @return The rotated image tensor with expanded resolution, or an
#'   \code{image_with_rotated_box} object when the input is
#'   \code{image_with_bounding_box}.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/6/66/The_Leaning_Tower_of_Pisa_SB.jpeg"
#'
#' img <- base_loader(url) |>
#'   transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(720, 620, 1900, 3700), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = {"Leaning Tower of Pisa"}))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_rotate(before, angle = 30)
#'
#' before_plot <- draw_bounding_boxes(before, colors = {"blue"}, width = 10)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)
#' tensor_image_browse(before_plot)
#' tensor_image_browse(after_plot)
#' }
#'
#' @family item_unitary_transforms
#'
#' @importFrom torch nnf_affine_grid nnf_grid_sample
#' @export
item_transform_rotate <- function(x, angle = 0) {
  UseMethod("item_transform_rotate", x)
}

#' @export
item_transform_rotate.torch_tensor <- function(x, angle = 0) {
  rotate_image_tensor(x, angle)
}

#' @export
item_transform_rotate.image_with_bounding_box <- function(x, angle = 0) {
  orig_h <- as.numeric(x$x$shape[length(x$x$shape) - 1])
  orig_w <- as.numeric(x$x$shape[length(x$x$shape)])

  rotated_img <- rotate_image_tensor(x$x, angle)
  new_h <- as.integer(rotated_img$shape[2])
  new_w <- as.integer(rotated_img$shape[3])

  dx <- (new_w - orig_w) / 2
  dy <- (new_h - orig_h) / 2

  shifted_boxes <- x$y$boxes$clone()
  if (shifted_boxes$size(1) > 0) {
    shifted_boxes[, 1] <- shifted_boxes[, 1] + dx
    shifted_boxes[, 3] <- shifted_boxes[, 3] + dx
    shifted_boxes[, 2] <- shifted_boxes[, 2] + dy
    shifted_boxes[, 4] <- shifted_boxes[, 4] + dy
  }

  x$x <- rotated_img
  x$y$boxes <- shifted_boxes
  x$y$image_height <- new_h
  x$y$image_width <- new_w

  target_transform_rotate_box(x, angle = angle)
}

rotate_image_tensor <- function(img, angle) {
  if (img$ndim == 3) img <- img$unsqueeze(1)

  img_shape <- img$shape
  C <- img_shape[[2]]
  H <- img_shape[[3]]
  W <- img_shape[[4]]

  theta_rad <- angle * pi / 180
  ct <- cos(theta_rad)
  st <- sin(theta_rad)
  if (abs(ct) < 1e-10) ct <- 0
  if (abs(st) < 1e-10) st <- 0

  new_W <- as.integer(ceiling(W * abs(ct) + H * abs(st)))
  new_H <- as.integer(ceiling(W * abs(st) + H * abs(ct)))

  a <- ct * new_W / W
  b <- st * new_H / W
  c <- -st * new_W / H
  d <- ct * new_H / H

  theta_mat <- torch_tensor(matrix(c(a, b, 0, c, d, 0), nrow = 2, byrow = TRUE),
                            dtype = torch_float32())$unsqueeze(1)

  grid <- nnf_affine_grid(theta_mat, size = c(1L, C, new_H, new_W), align_corners = FALSE)
  nnf_grid_sample(img, grid, mode = "bilinear", padding_mode = "zeros",
                  align_corners = FALSE)$squeeze(1)
}
