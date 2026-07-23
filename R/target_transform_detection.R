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
#' Converts bounding boxes of a detection target from
#' \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} (xyxy) format to
#' \eqn{(x_{min}, y_{min}, x_{max}, y_{max}, r)} (xyxyr) format, where
#' \eqn{r} is the rotation angle in degrees (counter-clockwise).
#' For axis-aligned boxes, \eqn{r = 0}.
#'
#' @param target A list representing the detection target, containing at least:
#'   \itemize{
#'     \item `boxes` — tensor of shape `(N, 4)` with bounding boxes in xyxy format
#'     \item `image_height` (optional) — original image height, used for
#'       clamping boxes to remain within image bounds
#'     \item `image_width` (optional) — original image width, used for
#'       clamping boxes to remain within image bounds
#'     \item Other fields (labels, etc.) are preserved unchanged
#'   }
#' @param angle (numeric): Rotation angle in degrees (counter-clockwise).
#'   Default is \code{0}.
#'
#' @return A list with the same structure as the input target, where
#'   `boxes` is a tensor of shape `(N, 5)` in xyxyr format.
#'   When applied to a dataset, returns the same dataset with its
#'   `.getitem` method modified to return rotated-box targets.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/6/66/The_Leaning_Tower_of_Pisa_SB.jpeg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(720, 620, 1900, 3700), ncol = 4), dtype = torch_float32())
#'
#' # Original boxes (blue, axis-aligned)
#' before_plot <- draw_bounding_boxes(img, boxes = boxes, colors = "blue", width = 10)
#'
#' # Transform boxes to xyxyr with 4 degree rotation
#' target <- list(
#'   boxes = boxes,
#'   image_height = img$shape[2],
#'   image_width = img$shape[3]
#' )
#' rotated_target <- target_transform_rotate_box(target, angle = 4)
#'
#' # Rotated boxes (red, drawn as polygons)
#' after_plot <- draw_bounding_boxes(img, boxes = rotated_target$boxes, colors = "red", width = 10)
#'
#' grid <- vision_make_grid(
#'   torch_stack(list(before_plot, after_plot))$to(torch_float32()),
#'   scale = TRUE
#' )
#' tensor_image_browse(grid)
#' }
#'
#' @family target_transforms_detection
#'
#' @export
target_transform_rotate_box <- function(target, angle = 0) {
  orig_boxes <- target$boxes
  cxcywh <- box_xyxy_to_cxcywh(orig_boxes)
  cx <- cxcywh[, 1]$unsqueeze(-1)
  cy <- cxcywh[, 2]$unsqueeze(-1)

  boxes <- box_xyxy_to_xyxyr(orig_boxes, angle = angle)

  img_h <- target$image_height
  img_w <- target$image_width

  if (!is.null(img_h) && !is.null(img_w)) {
    img_h <- as.numeric(img_h)
    img_w <- as.numeric(img_w)

    angle_col <- boxes[, 5]
    angle_rad <- torch_deg2rad(angle_col$reshape(c(-1, 1)))
    ct <- torch_cos(angle_rad)
    st <- torch_sin(angle_rad)

    hw <- (cxcywh[, 3] / 2)$unsqueeze(-1)
    hh <- (cxcywh[, 4] / 2)$unsqueeze(-1)

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

  target$boxes <- boxes
  target
}

#' @rdname target_transform_rotate_box
#' @export
target_transform_rotate_box.dataset <- function(target, angle = 0) {
  original_getitem <- target$.getitem
  target$.getitem <- function(index) {
    item <- original_getitem(index)
    item$y <- target_transform_rotate_box(item$y, angle = angle)
    item
  }
  target
}