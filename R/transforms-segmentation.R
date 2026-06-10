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
    y$masks <- torch::torch_stack(valid_masks)
  } else {
    y$masks <- torch::torch_zeros(c(0, y$image_height, y$image_width), dtype = torch::torch_bool())
  }

  y
}

parse_sahi_crop_window <- function(crop_window) {
  if (inherits(crop_window, "torch_tensor")) {
    crop_window <- as.array(crop_window)
  }

  if (is.matrix(crop_window) || is.data.frame(crop_window)) {
    crop_window <- as.numeric(crop_window)
  }

  if (is.numeric(crop_window) && length(crop_window) == 4L) {
    if (!is.null(names(crop_window))) {
      top <- crop_window[["top"]]
      left <- crop_window[["left"]]
      height <- crop_window[["height"]]
      width <- crop_window[["width"]]
    } else {
      top <- crop_window[[1]]
      left <- crop_window[[2]]
      height <- crop_window[[3]]
      width <- crop_window[[4]]
    }
  } else if (is.list(crop_window) && all(c("top", "left", "height", "width") %in% names(crop_window))) {
    top <- as.numeric(crop_window[["top"]])
    left <- as.numeric(crop_window[["left"]])
    height <- as.numeric(crop_window[["height"]])
    width <- as.numeric(crop_window[["width"]])
  } else {
    value_error(
      "Each crop window must be a numeric vector or list with elements ",
      "`top`, `left`, `height` and `width`."
    )
  }

  if (any(is.na(c(top, left, height, width)))) {
    value_error("Crop window must contain non-missing top, left, height and width values.")
  }
  if (top < 0 || left < 0) {
    value_error("Crop window top and left must be non-negative.")
  }
  if (height <= 0 || width <= 0) {
    value_error("Crop window height and width must be positive.")
  }

  list(
    top = as.numeric(top),
    left = as.numeric(left),
    height = as.numeric(height),
    width = as.numeric(width)
  )
}

normalize_sahi_crop_windows <- function(crop_windows) {
  if (inherits(crop_windows, "torch_tensor")) {
    crop_windows <- as.array(crop_windows)
  }

  if (is.numeric(crop_windows) && length(crop_windows) == 4L && !is.list(crop_windows)) {
    crop_windows <- list(crop_windows)
  }

  if (is.matrix(crop_windows)) {
    crop_windows <- lapply(seq_len(nrow(crop_windows)), function(i) crop_windows[i, ])
  } else if (is.data.frame(crop_windows)) {
    crop_windows <- lapply(seq_len(nrow(crop_windows)), function(i) as.numeric(crop_windows[i, ]))
  }

  if (is.list(crop_windows) && !is.null(names(crop_windows)) &&
      all(c("top", "left", "height", "width") %in% names(crop_windows))) {
    crop_windows <- list(crop_windows)
  }

  if (!is.list(crop_windows) || length(crop_windows) == 0L) {
    value_error("crop_windows must be a non-empty list of crop windows.")
  }

  lapply(crop_windows, parse_sahi_crop_window)
}

empty_like <- function(x) {
  if (inherits(x, "torch_tensor")) {
    return(torch::torch_zeros(c(0), dtype = x$dtype, device = x$device))
  }
  x[integer(0)]
}

select_object_field <- function(field, mask, mask_logical) {
  if (inherits(field, "torch_tensor")) {
    if (length(mask_logical) == 0L) {
      return(empty_like(field))
    }
    return(field[mask])
  }
  if (is.atomic(field) || is.list(field)) {
    return(field[mask_logical])
  }
  field
}

#' Target Transform: SAHI Crop Target Adjustment
#'
#' Adjusts object detection targets for SAHI (Slicing Aided Hyper Inference)
#' crop windows by clipping bounding boxes to crop boundaries, translating box
#' coordinates into the crop reference frame, and removing objects that do not
#' intersect the crop region.
#' Use as `target_transform` in object detection datasets together with SAHI
#' image slicing transforms.
#'
#' @param y list containing object detection target annotations with at least
#'   `boxes` and `labels` fields. Optional fields such as `area`, `iscrowd`,
#'   `image_height`, and `image_width` are preserved and updated when present.
#' @param crop_windows list of crop windows. Each crop window must define
#'   `top`, `left`, `height`, and `width`.
#' @param min_area numeric. Minimum bounding box area to retain after clipping.
#'   Objects with clipped area smaller than `min_area` are removed.
#'
#' @return List of transformed targets, one per crop window, with bounding boxes
#'   expressed in crop coordinates and associated annotations filtered to the
#'   objects visible within each crop.
#'
#' @examples
#' \dontrun{
#' target <- list(
#'   boxes = torch_tensor(
#'     matrix(
#'       c(
#'         10, 10, 30, 30,
#'         40, 40, 60, 60
#'       ),
#'       nrow = 2,
#'       byrow = TRUE
#'     )
#'   ),
#'   labels = torch_tensor(c(1L, 2L)),
#'   image_height = 100L,
#'   image_width = 100L
#' )
#'
#' crops <- list(
#'   list(top = 0, left = 0, height = 25, width = 25),
#'   list(top = 30, left = 30, height = 40, width = 40)
#' )
#'
#' cropped_targets <- target_transform_sahi_crop(
#'   target,
#'   crops
#' )
#' }
#'
#' @family target_transforms
#' @export
target_transform_sahi_crop <- function(y, crop_windows, min_area = 0) {
  if (!"boxes" %in% names(y)) {
    cli_abort("Target must contain 'boxes' field")
  }

  if (!"labels" %in% names(y)) {
    cli_abort("Target must contain 'labels' field")
  }

  if (!is.numeric(min_area) || length(min_area) != 1L || min_area < 0) {
    value_error("min_area must be a non-negative number.")
  }

  crop_windows <- normalize_sahi_crop_windows(crop_windows)

  boxes <- y$boxes

  if (!inherits(boxes, "torch_tensor")) {
    boxes <- torch::torch_tensor(
      boxes,
      dtype = torch::torch_float()
    )
  }

  if (boxes$ndim != 2L || boxes$shape[[2]] != 4L) {
    value_error("Target 'boxes' must be a tensor of shape (N, 4).")
  }

  labels <- y$labels

  if (inherits(labels, "torch_tensor") && labels$ndim == 0L) {
    labels <- labels$unsqueeze(1)
  }

  if ("area" %in% names(y) &&
      inherits(y$area, "torch_tensor") &&
      y$area$ndim == 0L) {
    y$area <- y$area$unsqueeze(1)
  }

  if ("iscrowd" %in% names(y) &&
      inherits(y$iscrowd, "torch_tensor") &&
      y$iscrowd$ndim == 0L) {
    y$iscrowd <- y$iscrowd$unsqueeze(1)
  }

  lapply(crop_windows, function(crop_window) {

    crop_top <- crop_window$top
    crop_left <- crop_window$left
    crop_bottom <- crop_top + crop_window$height
    crop_right <- crop_left + crop_window$width

    crop_top_t <- torch::torch_tensor(
      crop_top,
      dtype = boxes$dtype,
      device = boxes$device
    )

    crop_left_t <- torch::torch_tensor(
      crop_left,
      dtype = boxes$dtype,
      device = boxes$device
    )

    crop_bottom_t <- torch::torch_tensor(
      crop_bottom,
      dtype = boxes$dtype,
      device = boxes$device
    )

    crop_right_t <- torch::torch_tensor(
      crop_right,
      dtype = boxes$dtype,
      device = boxes$device
    )

    clipped_boxes <- torch::torch_stack(
      list(
        torch::torch_max(boxes[, 1], other = crop_left_t),
        torch::torch_max(boxes[, 2], other = crop_top_t),
        torch::torch_min(boxes[, 3], other = crop_right_t),
        torch::torch_min(boxes[, 4], other = crop_bottom_t)
      ),
      dim = 2
    )

    keep <- (
      clipped_boxes[, 3] > clipped_boxes[, 1]
    ) & (
      clipped_boxes[, 4] > clipped_boxes[, 2]
    )

    keep_mask <- as.logical(as.array(keep))

    if (any(keep_mask)) {
      new_boxes <- clipped_boxes[keep_mask, ]
    } else {
      new_boxes <- torch::torch_zeros(
        c(0, 4),
        dtype = boxes$dtype,
        device = boxes$device
      )
    }

    filtered_labels <- select_object_field(
      labels,
      keep,
      keep_mask
    )

    filtered_iscrowd <- NULL

    if ("iscrowd" %in% names(y)) {
      filtered_iscrowd <- select_object_field(
        y$iscrowd,
        keep,
        keep_mask
      )
    }

    if (as.integer(new_boxes$size(1)) > 0L) {

      offset <- torch::torch_tensor(
        c(
          crop_left,
          crop_top,
          crop_left,
          crop_top
        ),
        dtype = boxes$dtype,
        device = boxes$device
      )

      new_boxes <- new_boxes - offset
    }

    if (
      as.integer(new_boxes$size(1)) > 0L &&
      min_area > 0
    ) {

      area_mask <- as.logical(
        as.array(
          box_area(new_boxes) >= min_area
        )
      )

      if (any(area_mask)) {

        area_mask_t <- torch::torch_tensor(
          area_mask,
          dtype = torch::torch_bool(),
          device = boxes$device
        )

        new_boxes <- new_boxes[area_mask, ]

        filtered_labels <- select_object_field(
          filtered_labels,
          area_mask_t,
          area_mask
        )

        if (!is.null(filtered_iscrowd)) {
          filtered_iscrowd <- select_object_field(
            filtered_iscrowd,
            area_mask_t,
            area_mask
          )
        }

      } else {

        new_boxes <- torch::torch_zeros(
          c(0, 4),
          dtype = boxes$dtype,
          device = boxes$device
        )

        filtered_labels <- empty_like(labels)

        if (!is.null(filtered_iscrowd)) {
          filtered_iscrowd <- empty_like(y$iscrowd)
        }
      }
    }

    transformed <- y

    transformed$boxes <- new_boxes
    transformed$labels <- filtered_labels

    if ("area" %in% names(y)) {

      if (as.integer(new_boxes$size(1)) > 0L) {

        area <- box_area(new_boxes)

        if (inherits(y$area, "torch_tensor")) {
          area <- area$to(
            dtype = y$area$dtype,
            device = boxes$device
          )
        }

        transformed$area <- area

      } else {

        if (inherits(y$area, "torch_tensor")) {
          transformed$area <- torch::torch_zeros(
            c(0),
            dtype = y$area$dtype,
            device = boxes$device
          )
        } else {
          transformed$area <- numeric(0)
        }
      }
    }

    if (!is.null(filtered_iscrowd)) {
      transformed$iscrowd <- filtered_iscrowd
    }

    if ("image_height" %in% names(y)) {
      transformed$image_height <- as.integer(
        crop_window$height
      )
    }

    if ("image_width" %in% names(y)) {
      transformed$image_width <- as.integer(
        crop_window$width
      )
    }

    transformed
  })
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
    trimap <- torch::torch_tensor(trimap, dtype = torch::torch_int32())
  }

  if (trimap$ndim != 2) {
    cli::cli_abort("Trimap must be a 2D tensor")
  }

  mask1 <- (trimap == 1)
  mask2 <- (trimap == 2)
  mask3 <- (trimap == 3)

  y$masks <- torch::torch_stack(list(mask1, mask2, mask3))$to(dtype = torch::torch_bool())

  y
}
