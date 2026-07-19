#' Rotate image
#'
#' Rotates an image by a given angle around its center. The canvas is expanded
#' so that the entire rotated image is visible with no cropping. Empty regions
#' are filled with black.
#'
#' @param x A torch tensor of shape \code{(C, H, W)}.
#' @param angle (numeric): Rotation angle in degrees (counter-clockwise).
#'   Default is \code{0}.
#'
#' @return The rotated image tensor with expanded resolution.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/c/c6/Jumping_dog.JPG"
#'
#' img <- base_loader(url) |>
#'   transform_to_tensor() |>
#'   transform_resize(c(300, 500))
#'
#' before_plot <- draw_bounding_boxes(img, colors = "blue", width = 4)
#' after_plot <- draw_bounding_boxes(item_transform_rotate(img, angle = 30), colors = "red", width = 4)
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot))$to(torch_float32()), scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_unitary_transforms
#'
#' @importFrom torch nnf_affine_grid nnf_grid_sample
#' @export
item_transform_rotate <- function(x, angle = 0) {
  img <- x
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
