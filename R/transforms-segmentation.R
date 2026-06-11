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

#' Transform: SAHI Image Slicing
#'
#' Splits a large image into overlapping crops following the
#' SAHI (Slicing Aided Hyper Inference) approach. This transform is useful
#' for object detection workflows where downscaling large images would
#' otherwise remove small object details.
#'
#' The transform returns both the cropped image tiles and the crop window
#' metadata required to later transform detection targets using
#' `target_transform_sahi_crop()`.
#'
#' Use as `transform` in image datasets.
#'
#' @param x Image input. Can be a `torch_tensor` of shape `(C, H, W)` or a
#'   supported image type (e.g., `magick-image`, `array`). Non-tensor inputs
#'   are converted using `transform_to_tensor()` before slicing.
#' @note If `x` is not a `torch_tensor`, `transform_to_tensor()` is applied.
#' @param size Integer vector of length 2 containing crop height and width
#'   in the form `c(height, width)`.
#' @param overlap_size_ratio Numeric vector of length 2 containing vertical
#'   and horizontal overlap ratios in the form
#'   `c(overlap_height_ratio, overlap_width_ratio)`.
#'
#' @return A list with:
#' \describe{
#'   \item{images}{List of cropped image tensors.}
#'   \item{crop_windows}{List of crop windows. Each crop window contains
#'   `top`, `left`, `height` and `width` fields.}
#' }
#'
#' @examples
#' \dontrun{
#' ds <- mnist_dataset(
#'   transform = transform_sahi_crop,
#'   download = TRUE
#' )
#'
#' item <- ds[1]
#'
#' item$x$crop_windows[[1]]$top  # Top coordinate of the first crop in the original image
#'
#' item$x$crop_windows[[1]]$left  # Left coordinate of the first crop in the original image
#'
#' item$x$crop_windows[[1]]$height  # Height of the first crop
#'
#' item$x$crop_windows[[1]]$width  # Width of the first crop
#'
#' item$x$images[[1]] # First cropped image tensor
#'
#' item$y  # Dataset target (digit label)
#' }
#'
#' @references
#' SAHI Documentation:
#' https://obss.github.io/sahi/slicing/
#'
#' @family transforms
#' @export
transform_sahi_crop <- function(
  x,
  size = c(512L, 512L),
  overlap_size_ratio = c(0.2, 0.2)
){

  if (!inherits(x, "torch_tensor")) {
    x <- transform_to_tensor(x)
  }

  if (length(size) != 2L) {
    cli_abort("size must contain height and width.")
  }

  if (length(overlap_size_ratio) != 2L) {
    cli_abort("overlap_size_ratio must contain height and width ratios.")
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

#' Target Transform: SAHI Crop Target Adjustment
#'
#' Adjusts object detection targets for SAHI image crops by clipping bounding
#' boxes to crop boundaries, translating coordinates into crop-relative
#' coordinates and filtering annotations according to the retained area ratio.
#'
#' This transform is dataset-agnostic and operates on detection targets
#' containing at least `boxes` and `labels` fields. Crop windows are typically
#' generated by `transform_sahi_crop()`.
#'
#' Filtering follows the SAHI `min_area_ratio` definition:
#'
#' `cropped_annotation_area / original_annotation_area`
#'
#' Objects whose retained area ratio falls below `min_area_ratio`
#' are removed from the crop target.
#'
#' @param y Detection target list containing at least `boxes` and `labels`.
#'   Bounding boxes are expected in `(x1, y1, x2, y2)` format.
#' @param crop_windows List of crop windows generated by
#'   `transform_sahi_crop()`. Each crop window must contain `top`, `left`,
#'   `height` and `width`.
#' @param min_area_ratio Numeric value between 0 and 1 specifying the minimum
#'   fraction of original bounding box area that must remain after clipping
#'   for an annotation to be retained.
#'
#' @return A list of transformed targets, one per crop window. Each target
#' retains the original fields while updating:
#' \describe{
#'   \item{boxes}{Clipped and crop-relative bounding boxes.}
#'   \item{labels}{Labels corresponding to retained boxes.}
#'   \item{area}{Area of retained boxes when present in the original target.}
#'   \item{image_height}{Crop height when present in the original target.}
#'   \item{image_width}{Crop width when present in the original target.}
#' }
#'
#' @examples
#' \dontrun{
#' crop_result <- transform_sahi_crop(
#'   image,
#'   size = c(512, 512),
#'   overlap_size_ratio = c(0.2, 0.2)
#' )
#'
#' targets <- target_transform_sahi_crop(
#'   y,
#'   crop_result$crop_windows,
#'   min_area_ratio = 0.1
#' )
#'
#' length(targets)
#' targets[[1]]$boxes
#' }
#'
#' @references
#' SAHI Documentation:
#' https://obss.github.io/sahi/slicing/
#'
#' @family target_transforms
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
    cli_abort("min_area_ratio must be between 0 and 1.")
  }

  crop_windows <- normalize_sahi_crop_windows(crop_windows)

  boxes <- y$boxes

  if (!inherits(boxes, "torch_tensor")) {
    boxes <- torch_tensor(
      boxes,
      dtype = torch_float()
    )
  }

  if (
    boxes$ndim != 2L ||
    boxes$shape[[2]] != 4L
  ) {
    cli_abort("Target 'boxes' must have shape (N, 4).")
  }

  labels <- y$labels

  original_area <- box_area(boxes)

  lapply(crop_windows, function(window) {

    crop_top <- window$top
    crop_left <- window$left
    crop_bottom <- crop_top + window$height
    crop_right <- crop_left + window$width

    clipped_boxes <- torch_stack(
      list(
        torch_clamp(
          boxes[, 1],
          min = crop_left,
          max = crop_right
        ),
        torch_clamp(
          boxes[, 2],
          min = crop_top,
          max = crop_bottom
        ),
        torch_clamp(
          boxes[, 3],
          min = crop_left,
          max = crop_right
        ),
        torch_clamp(
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

    keep_mask <- as.logical(as.array(keep))

    if (any(keep_mask)) {

      clipped_boxes <- clipped_boxes[keep_mask,]

      original_area_kept <- original_area[keep_mask]

      labels_kept <- select_object_field(
        labels,
        torch_tensor(
          keep_mask,
          dtype = torch_bool()
        ),
        keep_mask
      )

      cropped_area <- box_area(clipped_boxes)

      area_ratio <- (cropped_area / original_area_kept)

      ratio_mask <- as.logical(
        as.array(
          area_ratio >= min_area_ratio
        )
      )

      clipped_boxes <- clipped_boxes[ratio_mask,]

      labels_out <- select_object_field(
        labels_kept,
        torch_tensor(
          ratio_mask,
          dtype = torch_bool()
        ),
        ratio_mask
      )

    } else {

      clipped_boxes <- torch_zeros(
        c(0, 4),
        dtype = boxes$dtype,
        device = boxes$device
      )

      labels_out <- empty_like(labels)
    }

    if (
      as.integer(
        clipped_boxes$size(1)
      ) > 0L
    ) {

      offset <- torch_tensor(
        c(
          crop_left,
          crop_top,
          crop_left,
          crop_top
        ),
        dtype = boxes$dtype,
        device = boxes$device
      )

      clipped_boxes <- (clipped_boxes - offset)
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

        transformed$area <- box_area(clipped_boxes)

      } else {

        transformed$area <- torch_zeros(
          c(0),
          dtype = boxes$dtype,
          device = boxes$device
        )
      }
    }

    if ("image_height" %in% names(y)) {
      transformed$image_height <- as.integer(window$height)
    }

    if ("image_width" %in% names(y)) {
      transformed$image_width <- as.integer(window$width)
    }

    transformed
  })
}

empty_like <- function(x) {

  if (inherits(x, "torch_tensor")) {

    return(
      torch_zeros(
        c(0),
        dtype = x$dtype,
        device = x$device
      )
    )
  }

  x[integer(0)]
}


select_object_field <- function(
    field,
    mask,
    mask_logical
) {

  if (inherits(field, "torch_tensor")) {

    if (length(mask_logical) == 0L) {
      return(empty_like(field))
    }

    return(field[mask])
  }

  if (is.atomic(field)) {
    return(field[mask_logical])
  }

  if (is.list(field)) {
    return(field[mask_logical])
  }

  field
}


parse_sahi_crop_window <- function(window) {

  if (
    !is.list(window) ||
    !all(
      c(
        "top",
        "left",
        "height",
        "width"
      ) %in% names(window)
    )
  ) {
    cli_abort("Each crop window must contain top, left, height and width.")
  }

  top <- as.numeric(window$top)
  left <- as.numeric(window$left)
  height <- as.numeric(window$height)
  width <- as.numeric(window$width)

  if (
    any(is.na(c(top, left, height, width)))
  ) {
    cli_abort("Crop window contains NA values.")
  }

  if (top < 0 || left < 0) {
    cli_abort("Crop window top and left must be non-negative.")
  }

  if (height <= 0 || width <= 0) {
    cli_abort("Crop window height and width must be positive.")
  }

  list(
    top = top,
    left = left,
    height = height,
    width = width
  )
}


normalize_sahi_crop_windows <- function(
    crop_windows
) {

  if (!is.list(crop_windows)) {
    cli_abort("crop_windows must be a list.")
  }

  if (length(crop_windows) == 0L) {
    cli_abort("crop_windows must not be empty.")
  }

  lapply(
    crop_windows,
    parse_sahi_crop_window
  )
}


box_area <- function(boxes) {

  (
    boxes[, 3] - boxes[, 1]
  ) *
  (
    boxes[, 4] - boxes[, 2]
  )
}