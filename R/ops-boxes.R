#' Non-maximum Suppression (NMS)
#'
#' Performs non-maximum suppression  (NMS) on the boxes according
#' to their intersection-over-union  (IoU). NMS iteratively removes
#' lower scoring boxes which have an IoU greater than iou_threshold
#' with another (higher scoring) box.
#'
#' @param boxes  (Tensor\[N, 4\])): boxes to perform NMS on. They are
#' expected to be in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format with
#' * \eqn{0 \leq x_{min} < x_{max}} and
#' * \eqn{0 \leq y_{min} < y_{max}}.
#' @param scores (Tensor\[N\]): scores for each one of the boxes
#' @param iou_threshold  (float): discards all overlapping boxes with IoU > iou_threshold
#'
#' @details
#'    If multiple boxes have the exact same score and satisfy the IoU
#'    criterion with respect to a reference box, the selected box is
#'    not guaranteed to be the same between CPU and GPU. This is similar
#'    to the behavior of argsort in torch when repeated values are present.
#'
#'    Current algorithm has a time complexity of O(n^2) and runs in native R.
#'    It may be improve in the future by a Rcpp implementation or through alternative algorithm
#'
#' @return keep (Tensor): int64 tensor with the indices of the elements that
#'  have been kept by NMS, sorted in decreasing order of scores.
#'
#' @export
nms <- function(boxes, scores, iou_threshold) {
  # assert_has_ops()
  # return(torch::torch_nms(boxes, scores, iou_threshold))
  if (length(scores) == 0) {
    return(integer())
  }

  # Sort scores in descending order
  order <- scores$sort(descending = TRUE)[[2]]
  boxes <- boxes[order, ]
  scores <- scores[order]

  keep <- c(1L)

  for (i in 2:length(scores)) {
    # Compute IoU with the last kept box
    iou <- box_iou(boxes[keep, , drop = FALSE], boxes[i, , drop = FALSE])
    # Check if the current box has IoU <= iou_threshold with all kept boxes
    if (all(as.logical(iou <= iou_threshold))) {
      keep <- c(keep, i)
    }
  }

  return(order[keep])
}


#' Batched Non-maximum Suppression (NMS)
#'
#'    Performs non-maximum suppression in a batched fashion.
#'    Each index value correspond to a category, and NMS
#'    will not be applied between elements of different categories.
#'
#' @param boxes (Tensor\[N, 4\]): boxes where NMS will be performed. They are expected to be
#'  in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format with
#'  *  \eqn{0 \leq x_{min} < x_{max}} and
#'  *  \eqn{0 \leq y_{min} < y_{max}}.
#' @param scores  (Tensor\[N\]): scores for each one of the boxes
#' @param idxs  (Tensor\[N\]): indices of the categories for each one of the boxes.
#' @param iou_threshold  (float): discards all overlapping boxes with IoU > `iou_threshold`
#'
#'
#' @return keep (Tensor): int64 tensor with the indices of
#' the elements that have been kept by NMS, sorted
#' in decreasing order of scores
#'
#' @export
batched_nms <- function(
  boxes,
  scores,
  idxs,
  iou_threshold
) {
  boxes_dtype = boxes$dtype
  boxes_device = boxes$device

  # strategy: in order to perform NMS independently per class.
  # we add an offset to all the boxes. The offset is dependent
  # only on the class idx, and is large enough so that boxes
  # from different classes do not overlap

    if(boxes$numel() == 0) {
      return(torch::torch_empty(0, dtype=torch::torch_int64(), device = boxes_device))
    } else {
      max_coordinate = boxes$max()
      offsets = idxs$to(device = boxes_device, dtype = boxes_dtype) * (max_coordinate + torch::torch_tensor(1)$to(device = boxes_device, dtype = boxes_dtype))
      boxes_for_nms = boxes + offsets[, NULL]
      keep = nms(boxes_for_nms, scores, iou_threshold)
      return(keep)
    }
}

#' Remove Small Boxes
#'
#' Remove boxes which contains at least one side smaller than min_size.
#'
#' @param boxes  (Tensor\[N, 4\]): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format
#'  with
#'  * \eqn{0 \leq x_{min} < x_{max}} and
#'  * \eqn{0 \leq y_{min} < y_{max}}.
#' @param min_size  (float): minimum size
#'
#' @return keep (Tensor\[K\]): indices of the boxes that have both sides
#'  larger than min_size
#'
#' @export
remove_small_boxes <- function(boxes, min_size) {
  c(ws, hs) %<-% c(boxes[, 3] - boxes[, 1], boxes[, 4] - boxes[, 2])
  keep = (ws >= min_size) & (hs >= min_size)
  keep = torch::torch_where(keep)[[1]]
  return(keep)
}

#' Clip Boxes to Image
#'
#' Clip boxes so that they lie inside an image of size `size`.
#'
#' @param boxes  (Tensor\[N, 4\]): boxes in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format
#' with
#' * \eqn{0 \leq x_{min} < x_{max}} and
#' * \eqn{0 \leq y_{min} < y_{max}}.
#' @param size  (Tuple\[height, width]): size of the image
#'
#' @return clipped_boxes (Tensor\[N, 4\])
#'
#' @export
clip_boxes_to_image <- function(boxes, size) {
  dim = boxes$dim()
  boxes_x = boxes[.., seq(1, boxes$shape[2], 2)]
  boxes_y = boxes[.., seq(2, boxes$shape[2], 2)]
  c(height, width) %<-% size

  # if(torchvision$_is_tracing()) {
  #   boxes_x = torch::torch_max(boxes_x, other = torch::torch_tensor(0, dtype=boxes$dtype, device=boxes$device))
  #   boxes_x = torch::torch_min(boxes_x, other = torch::torch_tensor(width, dtype=boxes$dtype, device=boxes$device))
  #   boxes_y = torch::torch_max(boxes_y, other = torch::torch_tensor(0, dtype=boxes$dtype, device=boxes$device))
  #   boxes_y = torch::torch_min(boxes_y, other = torch::torch_tensor(height, dtype=boxes$dtype, device=boxes$device))
  # } else {
  boxes_x = boxes_x$clamp(min=0, max=width)
  boxes_y = boxes_y$clamp(min=0, max=height)

  clipped_boxes = torch::torch_stack(c(boxes_x, boxes_y), dim=dim+1)
  return(clipped_boxes$reshape(boxes$shape))
}

#' Box Convert
#'
#'  Converts boxes from given in_fmt to out_fmt.
#'
#' @param boxes  (Tensor\[N, 4\]): boxes which will be converted.
#' @param in_fmt  (str): Input format of given boxes. Supported formats are \['xyxy', 'xywh', 'cxcywh'\].
#' @param out_fmt  (str): Output format of given boxes. Supported formats are \['xyxy', 'xywh', 'cxcywh'\]
#' @return boxes (Tensor\[N, 4]): Boxes into converted format.
#'
#' @details
#' Supported in_fmt and out_fmt are:
#' * 'xyxy': boxes are represented via corners,
#'    * \eqn{x_{min}, y_{min}} being top left and
#'    * \eqn{x_{max}, y_{max}} being bottom right.
#' * 'xywh' : boxes are represented via corner, width and height,
#'    * \eqn{x_{min}, y_{min}} being top left,
#'    * w, h being width and height.
#' * 'cxcywh' : boxes are represented via centre, width and height,
#'    * \eqn{c_x, c_y} being center of box,
#'    * w, h  being width and height.
#'
#' @export
box_convert <- function(boxes, in_fmt, out_fmt) {
  allowed_fmts = c("xyxy", "xywh", "cxcywh")
  if((!in_fmt %in% allowed_fmts) | (!out_fmt %in% allowed_fmts))
    value_error("Unsupported Bounding Box Conversions for given in_fmt and out_fmt")

  if(in_fmt == out_fmt)
    return(boxes$clone())

  if(in_fmt != 'xyxy' & out_fmt != 'xyxy') {
    # convert to xyxy and change in_fmt xyxy
    if(in_fmt == "xywh") {
      boxes = box_xywh_to_xyxy(boxes)
    } else if(in_fmt == "cxcywh") {
      boxes = box_cxcywh_to_xyxy(boxes)
    }
    in_fmt = 'xyxy'
  }
  if(in_fmt == "xyxy") {
    if(out_fmt == "xywh") {
      boxes = box_xyxy_to_xywh(boxes)
    } else if(out_fmt == "cxcywh") {
      boxes = box_xyxy_to_cxcywh(boxes)
    }
  } else if(out_fmt == "xyxy") {
    if(in_fmt == "xywh") {
      boxes = box_xywh_to_xyxy(boxes)
    } else if(in_fmt == "cxcywh") {
      boxes = box_cxcywh_to_xyxy(boxes)
    }
  }
  return(boxes)
}

upcast <- function(t) {
  t_dtype <- as.character(t$dtype)
  # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
  if(t$is_floating_point()) {
    return(if(t_dtype %in% c("Float", "Double")) t else t$to(device = torch::torch_float()))
  } else {
    return(if(t_dtype %in% c("Int", "Long")) t else t$to(device = torch::torch_int()))
  }
}

#' Box Area
#'
#' Computes the area of a set of bounding boxes, which are specified by its
#' \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} coordinates.
#'
#' @param boxes  (Tensor\[N, 4\]): boxes for which the area will be computed. They
#' are expected to be in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format with
#' * \eqn{0 \leq x_{min} < x_{max}} and
#' * \eqn{0 \leq y_{min} < y_{max}}.
#'
#' @return area (Tensor\[N\]): area for each box
#'
#' @export
box_area <- function(boxes) {
  boxes = upcast(boxes)
  return((boxes[, 3] - boxes[, 1]) * (boxes[, 4] - boxes[, 2]))
}

box_inter_union <- function(boxes1, boxes2) {
  # implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
  # with slight modifications
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)

  lt = torch::torch_max(boxes1[, NULL, 1:2], other = boxes2[, 1:2]) # [N,M,2]
  rb = torch::torch_min(boxes1[, NULL, 3:N], other = boxes2[, 3:N]) # [N,M,2]

  wh = upcast(rb - lt)$clamp(min=0) # [N,M,2]
  inter = wh[, , 1] * wh[, , 2] # [N,M]

  union = area1[, NULL] + area2 - inter

  return(list(inter, union))
}

#' Box IoU
#'
#' Return intersection-over-union  (Jaccard index) of boxes.
#' Both sets of boxes are expected to be in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format with
#' \eqn{0 \leq x_{min} < x_{max}} and \eqn{0 \leq y_{min} < y_{max}}.
#'
#' @param boxes1  (Tensor\[N, 4\])
#' @param boxes2  (Tensor\[M, 4\])
#'
#' @return iou (Tensor\[N, M\]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
#'
#' @export
box_iou <- function(boxes1, boxes2) {
    c(inter, union) %<-% box_inter_union(boxes1, boxes2)
    iou = inter / union
    return(iou)
}

#' Generalized Box IoU
#'
#' Return generalized intersection-over-union  (Jaccard index) of boxes.
#' Both sets of boxes are expected to be in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format with
#' \eqn{0 \leq x_{min} < x_{max}} and \eqn{0 \leq y_{min} < y_{max}}.
#'
#' @param boxes1  (Tensor\[N, 4\])
#' @param boxes2  (Tensor\[M, 4\])
#'
#' @details
#' Implementation adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
#'
#' @return generalized_iou (Tensor\[N, M\]): the NxM matrix containing the pairwise generalized_IoU values
#'        for every element in boxes1 and boxes2
#'
#' @export
generalized_box_iou <- function(boxes1, boxes2) {
  # degenerate boxes gives inf / nan results
  # so do an early check
  if(as.numeric((boxes1[, 3:N] >= boxes1[, 1:2])$all()) != 1)
    value_error("(boxes1[, 3:N] >= boxes1[, 1:2])$all() not TRUE")
  if(as.numeric((boxes2[, 3:N] >= boxes2[, 1:2])$all()) != 1)
    value_error("(boxes2[, 3:N] >= boxes2[, 1:2])$all() not TRUE")

  c(inter, union) %<-% box_inter_union(boxes1, boxes2)
  iou = inter / union

  lti = torch::torch_min(boxes1[, NULL, 1:2], other = boxes2[, 1:2])
  rbi = torch::torch_max(boxes1[, NULL, 3:N], other = boxes2[, 3:N])

  whi = upcast(rbi - lti)$clamp(min=0) # [N,M,2]
  areai = whi[, , 1] * whi[, , 2]

  return(iou - (areai - union) / areai)
}
