#' Target Transform: COCO Polygon Segmentation to Masks
#'
#' Converts COCO-style polygon segmentation annotations to boolean mask tensors.
#' Use as `target_transform` in `coco_detection_dataset()`.
#'
#' @param y List containing COCO target with `segmentation`, `image_height`, `image_width`
#'
#' @return Modified y list with added `masks` field (N, H, W) boolean tensor
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
#' @export
target_transform_coco_masks <- function(y) {

  if (!"segmentation" %in% names(y)) {
    cli::cli_abort("Target must contain 'segmentation' field")
  }
  if (!all(c("image_height", "image_width") %in% names(y))) {
    cli::cli_abort("Target must contain 'image_height' and 'image_width' fields")
  }

  H <- y$image_height
  W <- y$image_width

  masks_list <- lapply(y$segmentation, function(seg) {
    if (is.list(seg) && length(seg) > 0) {
      mask <- coco_polygon_to_mask(seg, height = H, width = W)
      if (inherits(mask, "torch_tensor") && mask$ndim == 2) return(mask)
    }
    NULL
  })

  valid_masks <- Filter(function(m) !is.null(m), masks_list)

  if (length(valid_masks) > 0) {
    y$masks <- torch::torch_stack(valid_masks)
  } else {
    y$masks <- torch::torch_zeros(c(0, H, W), dtype = torch::torch_bool())
  }

  y
}


#' Target Transform: Trimap to Boolean Masks
#'
#' Converts Oxford-IIIT Pet trimap (values 1,2,3) to 3-channel boolean masks.
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
