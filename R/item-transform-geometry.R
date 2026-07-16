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
#' url <- "https://upload.wikimedia.org/wikipedia/commons/c/c6/Jumping_dog.JPG"
#'
#' img <- base_loader(url) |>
#'   transform_to_tensor() |>
#'   transform_resize(c(300, 500))
#'
#' boxes <- torch_tensor(matrix(c(90, 25, 450, 290), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = {"dog"}, image_height = 300L, image_width = 500L))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- target_transform_rotate_box(before, angle = 30)
#'
#' before_plot <- draw_bounding_boxes(before, color = "blue", width = 4)
#' after_plot <- draw_bounding_boxes(after, color = "red", width = 4)
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot))$to(torch_float32()), scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
target_transform_rotate_box <- function(x, angle = 0) {
  UseMethod("target_transform_rotate_box", x)
}

#' @export
target_transform_rotate_box.image_with_bounding_box <- function(x, angle = 0) {
  img_h <- x$y$image_height
  img_w <- x$y$image_width

  orig_boxes <- x$y$boxes
  c(x1, y1, x2, y2) %<-% orig_boxes$unbind(-1)
  cx <- ((x1 + x2) / 2)$reshape(c(-1, 1))
  cy <- ((y1 + y2) / 2)$reshape(c(-1, 1))

  boxes <- box_xyxy_to_xyxyr(orig_boxes, angle = angle)

  if (!is.null(img_h) && !is.null(img_w)) {
    img_h <- as.numeric(img_h)
    img_w <- as.numeric(img_w)

    angle_col <- boxes[, 5]
    angle_rad <- angle_col$reshape(c(-1, 1)) * pi / 180
    ct <- torch_cos(angle_rad)
    st <- torch_sin(angle_rad)

    hw <- ((x2 - x1) / 2)$reshape(c(-1, 1))
    hh <- ((y2 - y1) / 2)$reshape(c(-1, 1))

    dx <- torch_cat(list(
      -hw * ct + hh * st,
       hw * ct + hh * st,
       hw * ct - hh * st,
      -hw * ct - hh * st
    ), dim = -1)

    dy <- torch_cat(list(
      -hw * st - hh * ct,
       hw * st - hh * ct,
       hw * st + hh * ct,
      -hw * st + hh * ct
    ), dim = -1)

    dist_left  <- torch_max(torch_clamp(-dx, min = 0), dim = -1)[[1]]$reshape(c(-1, 1))
    dist_right <- torch_max(torch_clamp(dx, min = 0), dim = -1)[[1]]$reshape(c(-1, 1))
    dist_down  <- torch_max(torch_clamp(-dy, min = 0), dim = -1)[[1]]$reshape(c(-1, 1))
    dist_up    <- torch_max(torch_clamp(dy, min = 0), dim = -1)[[1]]$reshape(c(-1, 1))

    eps <- 1e-8
    scale <- torch_min(torch_cat(list(
      cx / torch_clamp(dist_left, min = eps),
      (img_w - cx) / torch_clamp(dist_right, min = eps),
      cy / torch_clamp(dist_down, min = eps),
      (img_h - cy) / torch_clamp(dist_up, min = eps)
    ), dim = -1), dim = -1)[[1]]$reshape(c(-1, 1))
    scale <- torch_clamp(scale, min = 0, max = 1.0)

    hw <- hw * scale
    hh <- hh * scale

    boxes <- torch_cat(list(cx - hw, cy - hh, cx + hw, cy + hh, angle_col$reshape(c(-1, 1))), dim = -1L)
  }

  x$y$boxes <- boxes
  class(x) <- c("image_with_rotated_box", setdiff(class(x), "image_with_bounding_box"))
  x
}

#' @export
target_transform_rotate_box.dataset <- function(x, angle = 0) {
  original_getitem <- x$.getitem
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    target_transform_rotate_box(item, angle = angle)
  }
  x
}
