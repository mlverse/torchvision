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


#' @export 
transform_sahi_crop <- function(
    x,
    size = c(512L, 512L),
    overlap_size_ratio = c(0.2, 0.2)
) {

  if (length(size) != 2L) {
    value_error("size must contain height and width.")
  }

  if (length(overlap_size_ratio) != 2L) {
    value_error(
      "overlap_size_ratio must contain height and width ratios."
    )
  }

  slice_height <- as.integer(size[[1]])
  slice_width <- as.integer(size[[2]])

  overlap_height_ratio <- overlap_size_ratio[[1]]
  overlap_width_ratio <- overlap_size_ratio[[2]]

  image_height <- x$shape[[2]]
  image_width <- x$shape[[3]]

  step_y <- max(
    1L,
    floor(slice_height * (1 - overlap_height_ratio))
  )

  step_x <- max(
    1L,
    floor(slice_width * (1 - overlap_width_ratio))
  )

  tops <- unique(
    c(
      seq(
        0,
        max(0, image_height - slice_height),
        by = step_y
      ),
      max(0, image_height - slice_height)
    )
  )

  lefts <- unique(
    c(
      seq(
        0,
        max(0, image_width - slice_width),
        by = step_x
      ),
      max(0, image_width - slice_width)
    )
  )

  crop_windows <- list()
  crops <- list()

  idx <- 1L

  for (top in tops) {
    for (left in lefts) {

      height <- min(
        slice_height,
        image_height - top
      )

      width <- min(
        slice_width,
        image_width - left
      )

      crop_windows[[idx]] <- list(
        top = as.numeric(top),
        left = as.numeric(left),
        height = as.numeric(height),
        width = as.numeric(width)
      )

      crops[[idx]] <- x[
        ,
        (top + 1):(top + height),
        (left + 1):(left + width)
      ]

      idx <- idx + 1L
    }
  }

  list(
    images = crops,
    crop_windows = crop_windows
  )
}

#' @export
target_transform_sahi_crop <- function(
    y,
    crop_windows,
    min_area_ratio = 0.1
) {

  if (!"boxes" %in% names(y)) {
    cli_abort("Target must contain 'boxes' field.")
  }

  if (!"labels" %in% names(y)) {
    cli_abort("Target must contain 'labels' field.")
  }

  if (
    !is.numeric(min_area_ratio) ||
    length(min_area_ratio) != 1L ||
    min_area_ratio < 0 ||
    min_area_ratio > 1
  ) {
    value_error(
      "min_area_ratio must be between 0 and 1."
    )
  }

  crop_windows <- normalize_sahi_crop_windows(
    crop_windows
  )

  boxes <- y$boxes

  if (!inherits(boxes, "torch_tensor")) {
    boxes <- torch::torch_tensor(
      boxes,
      dtype = torch::torch_float()
    )
  }

  if (
    boxes$ndim != 2L ||
    boxes$shape[[2]] != 4L
  ) {
    value_error(
      "Target 'boxes' must have shape (N, 4)."
    )
  }

  labels <- y$labels

  original_area <- box_area(boxes)

  lapply(crop_windows, function(window) {

    crop_top <- window$top
    crop_left <- window$left
    crop_bottom <- crop_top + window$height
    crop_right <- crop_left + window$width

    clipped_boxes <- torch::torch_stack(
      list(
        torch::torch_clamp(
          boxes[, 1],
          min = crop_left,
          max = crop_right
        ),
        torch::torch_clamp(
          boxes[, 2],
          min = crop_top,
          max = crop_bottom
        ),
        torch::torch_clamp(
          boxes[, 3],
          min = crop_left,
          max = crop_right
        ),
        torch::torch_clamp(
          boxes[, 4],
          min = crop_top,
          max = crop_bottom
        )
      ),
      dim = 2
    )

    keep <- (
      clipped_boxes[, 3] >
      clipped_boxes[, 1]
    ) &
    (
      clipped_boxes[, 4] >
      clipped_boxes[, 2]
    )

    keep_mask <- as.logical(
      as.array(keep)
    )

    if (any(keep_mask)) {

      clipped_boxes <- clipped_boxes[
        keep_mask,
      ]

      original_area_kept <- original_area[
        keep_mask
      ]

      cropped_area <- box_area(
        clipped_boxes
      )

      area_ratio <- (
        cropped_area /
        original_area_kept
      )

      ratio_mask <- as.logical(
        as.array(
          area_ratio >= min_area_ratio
        )
      )

      clipped_boxes <- clipped_boxes[
        ratio_mask,
      ]

      labels_out <- select_object_field(
        labels,
        torch::torch_tensor(
          ratio_mask,
          dtype = torch::torch_bool()
        ),
        ratio_mask
      )

    } else {

      clipped_boxes <- torch::torch_zeros(
        c(0, 4),
        dtype = boxes$dtype,
        device = boxes$device
      )

      labels_out <- empty_like(
        labels
      )
    }

    if (
      as.integer(
        clipped_boxes$size(1)
      ) > 0L
    ) {

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

      clipped_boxes <- (
        clipped_boxes - offset
      )
    }

    transformed <- y

    transformed$boxes <- clipped_boxes
    transformed$labels <- labels_out

    if ("area" %in% names(y)) {

      if (
        as.integer(
          clipped_boxes$size(1)
        ) > 0L
      ) {

        transformed$area <- box_area(
          clipped_boxes
        )

      } else {

        transformed$area <- torch::torch_zeros(
          c(0),
          dtype = boxes$dtype,
          device = boxes$device
        )
      }
    }

    if ("image_height" %in% names(y)) {
      transformed$image_height <- as.integer(
        window$height
      )
    }

    if ("image_width" %in% names(y)) {
      transformed$image_width <- as.integer(
        window$width
      )
    }

    transformed
  })
}