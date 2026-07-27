#' @importFrom zeallot %<-%
#' @importFrom torch torch_cos torch_sin torch_min torch_abs torch_sign
NULL

#' box_cxcywh_to_xyxy
#'
#' Converts bounding boxes from  \eqn{(c_x, c_y, w, h)} format to \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format.
#'  \eqn{(c_x, c_y)} refers to center of bounding box
#'  (w, h) are width and height of bounding box
#'
#' @param boxes  (Tensor\[N, 4]): boxes in \eqn{(c_x, c_y, w, h)} format which will be converted.
#'
#' @return boxes (Tensor(N, 4)): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format.
box_cxcywh_to_xyxy <- function(boxes) {
  # We need to change all 4 of them so some temporary variable is needed.
  c(cx, cy, w, h) %<-% boxes$unbind(-1)
  x1 <- cx - 0.5 * w
  y1 <- cy - 0.5 * h
  x2 <- cx + 0.5 * w
  y2 <- cy + 0.5 * h

  boxes <- torch::torch_stack(list(x1, y1, x2, y2), dim = -1)

  return(boxes)
}

#' box_xyxy_to_cxcywh
#'
#' Converts bounding boxes from  \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format to \eqn{(c_x, c_y, w, h)} format.
#'  (x1, y1) refer to top left of bounding box
#'  (x2, y2) refer to bottom right of bounding box
#'
#' @param boxes  (Tensor\[N, 4\]): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format which will be converted.
#'
#' @return boxes (Tensor(N, 4)): boxes in \eqn{(c_x, c_y, w, h)} format.
box_xyxy_to_cxcywh <- function(boxes) {
  c(x1, y1, x2, y2) %<-% boxes$unbind(-1)
  cx <- (x1 + x2) / 2
  cy <- (y1 + y2) / 2
  w <- x2 - x1
  h <- y2 - y1

  boxes <- torch::torch_stack(list(cx, cy, w, h), dim = -1)

  return(boxes)
}

#' box_xywh_to_xyxy
#'
#' Converts bounding boxes from  (x, y, w, h) format to \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format.
#' (x, y) refers to top left of bouding box.
#' (w, h) refers to width and height of box.
#'
#' @param boxes  (Tensor\[N, 4\]): boxes in (x, y, w, h) which will be converted.
#'
#' @return boxes (Tensor\[N, 4\]): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format.
box_xywh_to_xyxy <- function(boxes) {
  c(x, y, w, h) %<-% boxes$unbind(-1)
  boxes <- torch::torch_stack(list(x, y, x + w, y + h), dim = -1)
  return(boxes)
}

#' box_xyxy_to_xywh
#'
#' Converts bounding boxes from  \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format to (x, y, w, h) format.
#' (x1, y1) refer to top left of bounding box
#' (x2, y2) refer to bottom right of bounding box
#'
#' @param boxes  (Tensor\[N, 4\]): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} which will be converted.
#'
#' @return boxes (Tensor\[N, 4\]): boxes in (x, y, w, h) format.
box_xyxy_to_xywh <- function(boxes) {
  c(x1, y1, x2, y2) %<-% boxes$unbind(-1)
  w <- x2 - x1 # x2 - x1
  h <- y2 - y1 # y2 - y1
  boxes <- torch::torch_stack(list(x1, y1, w, h), dim = -1)
  return(boxes)
}

#' box_xyxy_to_xyxyr
#'
#' Converts bounding boxes from \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format to
#' \eqn{(x_{min}, y_{min}, x_{max}, y_{max}, r_{deg})} format, where \eqn{r_{deg}} is the rotation
#' angle in degrees (anti-clockwise). Handles composition natively.
#'
#' @param boxes (Tensor\[N, 4\] or \[N, 5\]): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})}
#'  or \eqn{(x_{min}, y_{min}, x_{max}, y_{max}, r_{deg})} format, where \eqn{r_{deg}} format.
#' @param angle (numeric): Rotation angle in degrees (anti-clockwise).
#'   A single numeric value or a tensor of shape (N,). Default is 0.
#'
#' @return (Tensor\[N, 5\]): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max}, r_{deg})} format with accumulated angle.
#' @export
box_xyxy_to_xyxyr <- function(boxes, angle = 0) {
  n <- boxes$size(1)
  is_already_rotated <- boxes$size(2) == 5

  # Handle empty input early
  if (n == 0) {
    angle_t <- torch_zeros(0, 1, dtype = boxes$dtype)
    return(torch_cat(list(boxes, angle_t), dim = -1))
  }

  cx <- cy <- hw <- hh <- curr_angle <- NULL

  if (is_already_rotated) {
    # Input is already xyxyr: recover physical half-dimensions from the current AABB
    x1 <- boxes[, 1, drop = FALSE]
    y1 <- boxes[, 2, drop = FALSE]
    x2 <- boxes[, 3, drop = FALSE]
    y2 <- boxes[, 4, drop = FALSE]
    curr_angle <- boxes[, 5, drop = FALSE]

    cx <- (x1 + x2) * 0.5
    cy <- (y1 + y2) * 0.5
    hw_aabb <- (x2 - x1) * 0.5
    hh_aabb <- (y2 - y1) * 0.5

    # Invert the projection matrix to retrieve physical half-dimensions
    curr_rad <- deg2rad(curr_angle)
    c_abs <- torch_abs(torch_cos(curr_rad))
    s_abs <- torch_abs(torch_sin(curr_rad))

    # Determinant of the projection matrix: cos^2(theta) - sin^2(theta) = cos(2*theta)
    # The sign is needed for correct inversion (e.g., -1 at 90 degrees)
    det <- c_abs^2 - s_abs^2

    # Clamp away from zero to avoid division by zero at 45 degrees, preserving sign
    det <- torch_sign(det) * torch_clamp(torch_abs(det), min = 1e-6)

    hw <- (c_abs * hw_aabb - s_abs * hh_aabb) / det
    hh <- (-s_abs * hw_aabb + c_abs * hh_aabb) / det

    # Physical dimensions must be non-negative
    hw <- torch_clamp(hw, min = 0)
    hh <- torch_clamp(hh, min = 0)
  } else {
    # Standard xyxy input: physical dimensions equal AABB dimensions
    x1 <- boxes[, 1, drop = FALSE]
    y1 <- boxes[, 2, drop = FALSE]
    x2 <- boxes[, 3, drop = FALSE]
    y2 <- boxes[, 4, drop = FALSE]
    cx <- (x1 + x2) * 0.5
    cy <- (y1 + y2) * 0.5
    hw <- (x2 - x1) * 0.5
    hh <- (y2 - y1) * 0.5
    curr_angle <- torch_zeros_like(cx)
  }

  # Normalize and accumulate rotation angle
  angle_deg <- if (inherits(angle, "torch_tensor")) {
    angle$to(dtype = boxes$dtype)$reshape(c(-1, 1))
  } else {
    torch_tensor(angle, dtype = boxes$dtype)$reshape(c(-1, 1))
  }

  if (angle_deg$size(1) == 1 && n > 1) {
    angle_deg <- angle_deg$expand(c(n, 1))
  }

  total_angle <- curr_angle + angle_deg

  # Compute new AABB half-extents from physical dimensions and total angle
  total_rad <- deg2rad(total_angle)
  c_abs <- torch_abs(torch_cos(total_rad))
  s_abs <- torch_abs(torch_sin(total_rad))

  new_hw <- hw * c_abs + hh * s_abs
  new_hh <- hw * s_abs + hh * c_abs

  # Construct and return the xyxyr tensor
  torch_cat(
    list(
      cx - new_hw,
      cy - new_hh,
      cx + new_hw,
      cy + new_hh,
      total_angle
    ),
    dim = -1L
  )
}
