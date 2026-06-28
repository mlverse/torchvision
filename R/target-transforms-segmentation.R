#' Target Transform: COCO Polygon Segmentation to Masks
#'
#' Converts COCO-style polygon segmentation annotations from target `$segmentation` variable
#' into boolean mask tensors as target `$masks` variable in order to ease later-on visualisation
#' via `draw_segmentation_mask()`.
#' Use as `target_transform` in `coco_detection_dataset()`.
#'
#' @param y list being COCO dataset target variable, with names `segmentation`, `image_height`, `image_width`.
#'
#' @return Modified `y` list with added `masks` field (N, H, W) boolean tensor, N being the number of
#' classes.
#'
#' @examples
#' \dontrun{
#' ds <- coco_detection_dataset(
#'   root = "data",
#'   target_transform = target_transform_coco_masks
#' )
#' item <- ds[1]
#' draw_segmentation_masks(item)
#' }
#'
#' @family target_transforms
#' @export
target_transform_coco_masks <- function(y) {

  if (!"segmentation" %in% names(y)) {
    cli::cli_abort("Target must contain 'segmentation' field")
  }
  if (!all(c("image_height", "image_width") %in% names(y))) {
    cli::cli_abort("Target must contain 'image_height' and 'image_width' fields")
  }

  masks_list <- lapply(y$segmentation, function(seg) {
    if (is.list(seg) && length(seg) > 0) {
      mask <- coco_polygon_to_mask(seg, y$image_height, y$image_width)
      if (inherits(mask, "torch_tensor") && mask$ndim == 2) return(mask)
    }
    NULL
  })

  valid_masks <- Filter(function(m) !is.null(m), masks_list)

  if (length(valid_masks) > 0) {
    y$masks <- torch_stack(valid_masks)
  } else {
    y$masks <- torch_zeros(c(0, y$image_height, y$image_width), dtype = torch_bool())
  }

  y
}


#' Target Transform: Trimap to Boolean Masks
#'
#' Converts Oxford-IIIT Pet dataset target `$trimap` variable (values 1,2,3) into
#' 3-channel boolean masks tensors as target `$masks` variable in order to ease later-on visualisation
#' via `draw_segmentation_mask()`.
#' Use as `target_transform` in `oxfordiiitpet_segmentation_dataset()`.
#'
#' @param y List containing `trimap` field (H, W) tensor with values 1, 2, 3
#'
#' @return Modified y list with added `masks` field (3, H, W) boolean tensor
#'
#' @details
#' Creates three mutually exclusive masks:
#' \itemize{
#'   \item Channel 1: Pet pixels (trimap == 1)
#'   \item Channel 2: Background pixels (trimap == 2)
#'   \item Channel 3: Outline pixels (trimap == 3)
#' }
#'
#' @examples
#' \dontrun{
#' ds <- oxfordiiitpet_segmentation_dataset(
#'   root = "data",
#'   target_transform = target_transform_trimap_masks
#' )
#' item <- ds[1]
#' draw_segmentation_masks(item)
#' }
#'
#' @family target_transforms
#' @export
target_transform_trimap_masks <- function(y) {
  if (!"trimap" %in% names(y)) {
    cli::cli_abort("Target must contain 'trimap' field")
  }

  trimap <- y$trimap

  if (!inherits(trimap, "torch_tensor")) {
    trimap <- torch_tensor(trimap, dtype = torch_int32())
  }

  if (trimap$ndim != 2) {
    cli::cli_abort("Trimap must be a 2D tensor")
  }

  mask1 <- (trimap == 1)
  mask2 <- (trimap == 2)
  mask3 <- (trimap == 3)

  y$masks <- torch_stack(list(mask1, mask2, mask3))$to(dtype = torch_bool())

  y
}

#' @rdname transform_sahi_crop
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
    out$is_crowd <- NULL
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

    out_y$is_crowd <- NULL

    out_y
  })

  results
}