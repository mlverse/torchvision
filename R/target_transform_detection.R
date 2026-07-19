#' Resize the bounding boxes of a detection target
#'
#' Adjusts the bounding box coordinates in a detection target to match a
#'   resized image. The target is expected to contain a `boxes` field with
#'   shape `(N, 4)` in xyxy format and an `orig_size` field with the original
#'   image dimensions `(H, W)`.
#'
#' @param target A list containing at least:
#'   \itemize{
#'     \item `boxes` — tensor of shape `(N, 4)` with bounding boxes in xyxy format
#'     \item `orig_size` — integer vector or tensor of length 2 with `(height, width)`
#'     \item Other fields (labels, image_id, etc.) are preserved unchanged
#'   }
#' @param size Desired output size. If `size` is a integer vector of length 2
#'   like `c(h, w)`, output size will be matched to this. If `size` is a bare integer,
#'   smaller edge of the image will be matched to this number.
#'   i.e, if height > width, then image will be rescaled to
#'   `(size * height / width, size)`.
#'
#' @return A list with the same structure as the input target, where:
#'   \itemize{
#'     \item `boxes` are rescaled according to the resize factors
#'     \item `orig_size` is updated to the new dimensions
#'   }
#'
#' @details
#' This function should be composed with a [transform_resize()] in the
#'   `transform` pipeline having the same `size` to ensure resized bounding boxes
#'  remain aligned with the resized image.
#'
#' @examples
#' \dontrun{
#' target <- list(
#'   boxes = torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4)),
#'   labels = torch_tensor(1L, dtype = torch_long()),
#'   image_height  = 100L,
#'   image_width = 100L)
#' )
#'
#' # Resize to fixed size
#' transform_fn <- target_transform_resize(c(200L, 200L))
#' new_target <- transform_fn(target)
#'
#' # Resize with proportional scaling
#' transform_fn <- target_transform_resize(50L)
#' new_target <- transform_fn(target)
#' }
#'
#' @family target_transforms_detection
#'
#' @export
target_transform_resize <- function(target, size) {
  if (!"image_height" %in% names(target) || !"image_width" %in% names(target)) {
    cli::cli_abort("Target must contain both 'image_height' and 'image_width' field")
  }
  # Extract original dimensions
  orig_h <- target$image_height
  orig_w <- target$image_width

  # Compute new dimensions
  if (length(size) == 1L) {
    # Proportional resize: match smaller edge to size
    scale <- size / max(orig_h, orig_w)
    new_h <- round(orig_h * scale)
    new_w <- round(orig_w * scale)
  } else {
    # Fixed size resize
    c(new_h, new_w) %<-% size
  }

  # Compute scale factors (reuse orig_h / orig_w)
  scale_h <- new_h / orig_h
  scale_w <- new_w / orig_w

  # Resize bounding boxes (xyxy format)
  boxes <- target$boxes$clone()
  boxes_resized <- boxes
  boxes_resized[, 1L] <- boxes[, 1L] * scale_w # xmin
  boxes_resized[, 2L] <- boxes[, 2L] * scale_h # ymin
  boxes_resized[, 3L] <- boxes[, 3L] * scale_w # xmax
  boxes_resized[, 4L] <- boxes[, 4L] * scale_h # ymax

  # Update target
  target$boxes <- boxes_resized
  target$image_height <- new_h
  target$image_width <- new_w

  target
}


#' @rdname transform_sahi_crop
#' @family target_transforms_detection
#' @export
target_transform_sahi_crop <- function(y, sahi_split, min_area_ratio = 0.1) {

  # Detect batch input: list of target lists
  if (is.list(y) && !"boxes" %in% names(y)) {
    if (is.list(sahi_split) && !inherits(sahi_split, "sahi_split")) {
      return(Map(function(yi, sp) target_transform_sahi_crop(yi, sp, min_area_ratio),
                 y, sahi_split))
    }
    return(lapply(y, function(yi) target_transform_sahi_crop(yi, sahi_split, min_area_ratio)))
  }

  boxes <- y$boxes
  labels <- y$labels
  labels_is_tensor <- inherits(labels, "torch_tensor")

  n <- boxes$size(1)

  crop_windows <- sahi_split$crop_windows

  shape_y <- function(n_boxes, boxes_dtype, labels_val, labels_dtype, crop_h, crop_w) {
    out <- y
    out$boxes <- torch_zeros(c(n_boxes, 4), dtype = boxes_dtype)
    if (labels_is_tensor)
      out$labels <- torch_tensor(labels_val, dtype = labels_dtype)
    else
      out$labels <- labels_val
    if (!is.null(y$area))
      out$area <- torch_zeros(n_boxes, dtype = y$area$dtype)
    if (!is.null(y$image_height))
      out$image_height <- crop_h
    if (!is.null(y$image_width))
      out$image_width <- crop_w
    out$iscrowd <- NULL
    out
  }

  results <- lapply(crop_windows, function(cw) {

    # Convert 1-based crop window coordinates to 0-based for box clipping
    top <- cw$top - 1
    left <- cw$left - 1
    crop_h <- cw$height
    crop_w <- cw$width

    if (n == 0) {
      labels_val <- if (labels_is_tensor) integer(0) else vector(typeof(labels), 0)
      labels_dtype <- if (labels_is_tensor) labels$dtype else NULL
      return(shape_y(0, boxes$dtype, labels_val, labels_dtype, crop_h, crop_w))
    }

    x1 <- boxes[, 1]
    y1 <- boxes[, 2]
    x2 <- boxes[, 3]
    y2 <- boxes[, 4]

    orig_area <- (x2 - x1) * (y2 - y1)

    x1_clip <- torch_clamp(x1, min = left, max = left + crop_w)
    y1_clip <- torch_clamp(y1, min = top, max = top + crop_h)
    x2_clip <- torch_clamp(x2, min = left, max = left + crop_w)
    y2_clip <- torch_clamp(y2, min = top, max = top + crop_h)

    keep_w <- x2_clip - x1_clip
    keep_h <- y2_clip - y1_clip
    keep_area <- keep_w * keep_h

    mask <- (keep_area > 0) & ((keep_area / orig_area) >= min_area_ratio)

    mask_idx <- which(as.logical(mask))

    if (length(mask_idx) == 0) {
      labels_val <- if (labels_is_tensor) integer(0) else vector(typeof(labels), 0)
      labels_dtype <- if (labels_is_tensor) labels$dtype else NULL
      return(shape_y(0, boxes$dtype, labels_val, labels_dtype, crop_h, crop_w))
    }

    n_keep <- length(mask_idx)
    new_boxes <- torch_zeros(n_keep, 4, dtype = boxes$dtype)
    new_boxes[, 1] <- x1_clip[mask_idx] - left
    new_boxes[, 2] <- y1_clip[mask_idx] - top
    new_boxes[, 3] <- x2_clip[mask_idx] - left
    new_boxes[, 4] <- y2_clip[mask_idx] - top

    out_y <- y
    out_y$boxes <- new_boxes

    if (labels_is_tensor)
      out_y$labels <- labels[mask_idx]
    else
      out_y$labels <- labels[mask_idx]

    if (!is.null(y$area))
      out_y$area <- keep_area[mask_idx]

    if (!is.null(y$image_height))
      out_y$image_height <- crop_h
    if (!is.null(y$image_width))
      out_y$image_width <- crop_w

    out_y$iscrowd <- y$iscrowd[mask_idx]

    out_y
  })

  results
}

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
#' after <- target_transform_rotate_box(before, angle = 4)
#'
#' before_plot <- draw_bounding_boxes(before, colors = {"blue"}, width = 40)
#' after_plot <- draw_bounding_boxes(after, colors = {"red"}, width = 40)
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
  if (!is.null(x$x) && inherits(x$x, "torch_tensor") && x$x$ndim >= 3) {
    img_h <- as.numeric(x$x$shape[length(x$x$shape) - 1])
    img_w <- as.numeric(x$x$shape[length(x$x$shape)])
  } else {
    img_h <- x$y$image_height
    img_w <- x$y$image_width
  }

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