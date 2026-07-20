#' @importFrom zeallot %<-%
#' @importFrom torch torch_cos torch_sin torch_min
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

  boxes <- torch::torch_stack(list(x1, y1, x2, y2), dim=-1)

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

  boxes <- torch::torch_stack(list(cx, cy, w, h), dim=-1)

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
  boxes <- torch::torch_stack(list(x, y, x + w, y + h), dim=-1)
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
  boxes <- torch::torch_stack(list(x1, y1, w, h), dim=-1)
  return(boxes)
}

#' box_xyxy_to_xyxyr
#'
#' Converts bounding boxes from \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format to
#'   \eqn{(x_{min}, y_{min}, x_{max}, y_{max}, r)} format, where \eqn{r} is the rotation
#'   angle in degrees (anti-clockwise). For axis-aligned boxes, \eqn{r = 0}.
#'
#' @param boxes (Tensor\[N, 4\]): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format
#'   which will be converted.
#' @param angle (numeric, optional): Rotation angle in degrees (anti-clockwise).
#'   A single numeric value applied to all boxes, or a tensor of shape \code{(N,)}
#'   with one angle per box. Default is \code{0}.
#'
#' @return boxes (Tensor\[N, 5\]): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max}, r)} format,
#'   where \eqn{r} is the provided rotation angle in degrees. The bounding box
#'   coordinates are computed by rotating the original axis-aligned box around
#'   its center by \eqn{r} degrees anti-clockwise, then taking the axis-aligned
#'   bounding box of the rotated corners.
#'
#' @export
box_xyxy_to_xyxyr <- function(boxes, angle = 0) {

  n <- boxes$size(1)

  if (n == 0) {
    angle_t <- torch_zeros(0, 1, dtype = boxes$dtype)
    return(torch_cat(list(boxes, angle_t), dim = -1))
  }

  c(x1, y1, x2, y2) %<-% boxes$unbind(-1)
  cx <- ((x1 + x2) / 2)$reshape(c(-1, 1))
  cy <- ((y1 + y2) / 2)$reshape(c(-1, 1))
  hw <- ((x2 - x1) / 2)$reshape(c(-1, 1))
  hh <- ((y2 - y1) / 2)$reshape(c(-1, 1))

  if (inherits(angle, "torch_tensor")) {
    angle_deg <- angle$to(dtype = boxes$dtype)$reshape(c(-1, 1))
  } else {
    angle_deg <- torch_tensor(angle, dtype = boxes$dtype)$reshape(c(-1, 1))
  }

  if (angle_deg$size(1) == 1 && n > 1) {
    angle_deg <- angle_deg$expand(c(n, 1))
  }

  angle_rad <- angle_deg * pi / 180
  ct <- torch_cos(angle_rad)
  st <- torch_sin(angle_rad)

  corners_x <- torch_cat(list(
    cx - hw * ct + hh * st,
    cx + hw * ct + hh * st,
    cx + hw * ct - hh * st,
    cx - hw * ct - hh * st
  ), dim = -1)

  corners_y <- torch_cat(list(
    cy - hw * st - hh * ct,
    cy + hw * st - hh * ct,
    cy + hw * st + hh * ct,
    cy - hw * st + hh * ct
  ), dim = -1)

  xmin <- torch_min(corners_x, dim = -1)[[1]]$reshape(c(-1, 1))
  xmax <- torch_max(corners_x, dim = -1)[[1]]$reshape(c(-1, 1))
  ymin <- torch_min(corners_y, dim = -1)[[1]]$reshape(c(-1, 1))
  ymax <- torch_max(corners_y, dim = -1)[[1]]$reshape(c(-1, 1))

  torch_cat(list(xmin, ymin, xmax, ymax, angle_deg), dim = -1L)
}

