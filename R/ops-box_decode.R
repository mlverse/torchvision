#' Apply box deltas to anchors
#'
#' Applies [dx, dy, dw, dh] deltas to anchor boxes to get predicted boxes.
#'
#' @param anchors  Tensor of shape (N, 4), in [x1, y1, x2, y2] format
#' @param deltas   Tensor of shape (N, 4), each row is [dx, dy, dw, dh]
#'
#' @return Tensor of shape (N, 4) with predicted boxes in [x1, y1, x2, y2] format
#' @export
apply_deltas_to_anchors <- function(anchors, deltas) {
  widths  <- anchors[, 3] - anchors[, 1]
  heights <- anchors[, 4] - anchors[, 2]
  ctr_x   <- anchors[, 1] + 0.5 * widths
  ctr_y   <- anchors[, 2] + 0.5 * heights

  dx <- deltas[, 1]
  dy <- deltas[, 2]
  dw <- deltas[, 3]
  dh <- deltas[, 4]

  pred_ctr_x <- dx * widths + ctr_x
  pred_ctr_y <- dy * heights + ctr_y
  pred_w     <- torch::torch_exp(dw) * widths
  pred_h     <- torch::torch_exp(dh) * heights

  pred_boxes_x1 <- pred_ctr_x - 0.5 * pred_w
  pred_boxes_y1 <- pred_ctr_y - 0.5 * pred_h
  pred_boxes_x2 <- pred_ctr_x + 0.5 * pred_w
  pred_boxes_y2 <- pred_ctr_y + 0.5 * pred_h

  torch::torch_stack(list(
    pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2
  ), dim = 2)
}
