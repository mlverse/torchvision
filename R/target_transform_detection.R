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
#'   orig_size = c(100L, 100L)
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
  # Extract original dimensions once
  c(orig_h, orig_w) %<-% target$orig_size

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
  boxes <- target$boxes
  boxes_resized <- boxes
  boxes_resized[, 1L] <- boxes[, 1L] * scale_w  # xmin
  boxes_resized[, 2L] <- boxes[, 2L] * scale_h  # ymin
  boxes_resized[, 3L] <- boxes[, 3L] * scale_w  # xmax
  boxes_resized[, 4L] <- boxes[, 4L] * scale_h  # ymax

  # Update target
  target$boxes     <- boxes_resized
  target$orig_size <- c(new_h, new_w)

  target
}
