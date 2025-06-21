#' @importFrom zeallot %<-%
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
  x1 = cx - 0.5 * w
  y1 = cy - 0.5 * h
  x2 = cx + 0.5 * w
  y2 = cy + 0.5 * h

  boxes = torch::torch_stack(list(x1, y1, x2, y2), dim=-1)

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
  cx = (x1 + x2) / 2
  cy = (y1 + y2) / 2
  w = x2 - x1
  h = y2 - y1

  boxes = torch::torch_stack(list(cx, cy, w, h), dim=-1)

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
  boxes = torch::torch_stack(list(x, y, x + w, y + h), dim=-1)
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
  w = x2 - x1 # x2 - x1
  h = y2 - y1 # y2 - y1
  boxes = torch::torch_stack(list(x1, y1, w, h), dim=-1)
  return(boxes)
}
