rpn_head <- torch::nn_module(
    "rpn_head",
    initialize = function(in_channels, num_anchors = 3) {
      self$conv <- nn_sequential(nn_sequential(nn_conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)))
      self$cls_logits <- nn_conv2d(in_channels, num_anchors, kernel_size = 1)
      self$bbox_pred <- nn_conv2d(in_channels, num_anchors * 4, kernel_size = 1)
    },
    forward = function(features) {
      objectness_list <- vector("list", length(features))
      bbox_reg_list <- vector("list", length(features))

      for (i in seq_along(features)) {
        x <- features[[i]]
        t <- torch::nnf_relu(self$conv(x))
        objectness_list[[i]] <- self$cls_logits(t)
        bbox_reg_list[[i]] <- self$bbox_pred(t)
      }

      list(objectness = objectness_list, bbox_deltas = bbox_reg_list)
    }
  )


rpn_head_v2 <- torch::nn_module(
    "rpn_head_v2",
    initialize = function(in_channels, num_anchors = 3) {
      # PyTorch V2 uses Conv2dNormActivation with norm_layer=None, which creates
      # Conv2d with bias=True, NO batch norm, and ReLU activation.
      # This matches torchvision.ops.misc.Conv2dNormActivation behavior when norm_layer=None.
      block <- function() {
        nn_sequential(
          nn_conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, bias = TRUE),
          nn_relu()
        )
      }
      self$conv <- nn_sequential(block(), block())
      self$cls_logits <- nn_conv2d(in_channels, num_anchors, kernel_size = 1, bias = TRUE)
      self$bbox_pred <- nn_conv2d(in_channels, num_anchors * 4, kernel_size = 1, bias = TRUE)
    },
    forward = function(features) {
      objectness_list <- vector("list", length(features))
      bbox_reg_list <- vector("list", length(features))

      for (i in seq_along(features)) {
        x <- features[[i]]
        t <- self$conv(x)
        objectness_list[[i]] <- self$cls_logits(t)
        bbox_reg_list[[i]] <- self$bbox_pred(t)
      }

      list(objectness = objectness_list, bbox_deltas = bbox_reg_list)
    }
  )


rpn_head_mobilenet <- torch::nn_module(
    initialize = function(in_channels, num_anchors = 15) {
      self$conv <- nn_sequential(nn_sequential(nn_conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)))
      self$cls_logits <- nn_conv2d(in_channels, num_anchors, kernel_size = 1)
      self$bbox_pred <- nn_conv2d(in_channels, num_anchors * 4, kernel_size = 1)
    },
    forward = function(features) {
      objectness <- vector("list", length(features))
      bbox_deltas <- vector("list", length(features))

      for (i in seq_along(features)) {
        t <- nnf_relu(self$conv(features[[i]]))
        objectness[[i]] <- self$cls_logits(t)
        bbox_deltas[[i]] <- self$bbox_pred(t)
      }

      list(objectness = objectness, bbox_deltas = bbox_deltas)
    }
  )


#' @importFrom torch torch_meshgrid torch_stack torch_tensor torch_stack torch_zeros_like torch_max torch_float32 torch_empty
NULL

# ---------------------------------------------------------------------------
# Generalized R-CNN building blocks, following torchvision.models.detection
# (AnchorGenerator, RegionProposalNetwork, BoxCoder, MultiScaleRoIAlign,
# RoIHeads.postprocess_detections, paste_masks_in_image).
# ---------------------------------------------------------------------------

# Index along the first dimension without dropping dimensions.
.isel <- function(x, idx) x$index_select(1, idx)

# Base anchors of one level, centred on (0, 0), ordered ratio-major / size-minor
# (torchvision AnchorGenerator.generate_anchors). aspect ratio = height / width.
rcnn_base_anchors <- function(scales, aspect_ratios, device = "cpu") {
  scales <- torch_tensor(scales, dtype = torch_float32(), device = device)
  aspect_ratios <- torch_tensor(aspect_ratios, dtype = torch_float32(), device = device)
  h_ratios <- torch::torch_sqrt(aspect_ratios)
  w_ratios <- 1 / h_ratios
  ws <- (w_ratios$unsqueeze(2) * scales$unsqueeze(1))$reshape(-1)
  hs <- (h_ratios$unsqueeze(2) * scales$unsqueeze(1))$reshape(-1)
  base <- torch_stack(list(-ws, -hs, ws, hs), dim = 2) / 2
  base$round()
}

# Anchors for every level, as a list of [H*W*A, 4] tensors in (H, W, A) order.
# Strides are image_size %/% feature_size per dimension, grid offsets start at 0.
rcnn_anchors <- function(image_size, feature_shapes, sizes, aspect_ratios, device = "cpu") {
  lapply(seq_along(feature_shapes), function(l) {
    fh <- feature_shapes[[l]][1]; fw <- feature_shapes[[l]][2]
    stride_h <- image_size[1] %/% fh; stride_w <- image_size[2] %/% fw
    base <- rcnn_base_anchors(sizes[[l]], aspect_ratios[[l]], device)
    shifts_x <- torch::torch_arange(0, fw - 1, dtype = torch::torch_int32(), device = device) * stride_w
    shifts_y <- torch::torch_arange(0, fh - 1, dtype = torch::torch_int32(), device = device) * stride_h
    sh <- torch_meshgrid(list(shifts_y, shifts_x), indexing = "ij")
    sy <- sh[[1]]$reshape(-1); sx <- sh[[2]]$reshape(-1)
    shifts <- torch_stack(list(sx, sy, sx, sy), dim = 2)
    (shifts$view(c(-1, 1, 4)) + base$view(c(1, -1, 4)))$reshape(c(-1, 4))
  })
}

# torchvision BoxCoder.decode_single. rel_codes [N, K*4], boxes [N, 4] -> [N, K, 4]
rcnn_decode <- function(rel_codes, boxes, weights = c(1, 1, 1, 1),
                        bbox_xform_clip = log(1000 / 16)) {
  boxes <- boxes$to(dtype = rel_codes$dtype)
  n <- rel_codes$shape[1]
  rel <- rel_codes$reshape(c(n, -1, 4))
  widths <- (boxes[, 3] - boxes[, 1])$unsqueeze(2)
  heights <- (boxes[, 4] - boxes[, 2])$unsqueeze(2)
  ctr_x <- boxes[, 1]$unsqueeze(2) + 0.5 * widths
  ctr_y <- boxes[, 2]$unsqueeze(2) + 0.5 * heights
  dx <- rel[, , 1] / weights[1]
  dy <- rel[, , 2] / weights[2]
  dw <- torch::torch_clamp(rel[, , 3] / weights[3], max = bbox_xform_clip)
  dh <- torch::torch_clamp(rel[, , 4] / weights[4], max = bbox_xform_clip)
  pred_ctr_x <- dx * widths + ctr_x
  pred_ctr_y <- dy * heights + ctr_y
  pred_w <- torch::torch_exp(dw) * widths
  pred_h <- torch::torch_exp(dh) * heights
  c_to_c_w <- 0.5 * pred_w
  c_to_c_h <- 0.5 * pred_h
  torch_stack(list(pred_ctr_x - c_to_c_w, pred_ctr_y - c_to_c_h,
                   pred_ctr_x + c_to_c_w, pred_ctr_y + c_to_c_h), dim = 3)
}

# Kept for backward compatibility: Fast R-CNN box decoding of [N, 4] deltas.
decode_boxes <- function(anchors, deltas, weights = c(10, 10, 5, 5)) {
  rcnn_decode(deltas, anchors, weights)$reshape(c(-1, 4))
}

# torchvision RegionProposalNetwork.filter_proposals (inference path), per image.
# objectness / bbox_deltas: lists of per-level head outputs [B, A, H, W] / [B, A*4, H, W]
# anchors: list of per-level [H*W*A, 4]. Returns a list (per image) of list(boxes, scores).
rcnn_filter_proposals <- function(objectness, bbox_deltas, anchors, image_size,
                                  pre_nms_top_n = 1000, post_nms_top_n = 1000,
                                  nms_thresh = 0.7, score_thresh = 0, min_size = 1e-3) {
  batch_size <- objectness[[1]]$shape[1]
  device <- objectness[[1]]$device
  lapply(seq_len(batch_size), function(b) {
    obj_l <- list(); box_l <- list(); lvl_l <- list()
    for (l in seq_along(objectness)) {
      o <- objectness[[l]][b, , , , drop = FALSE]
      d <- bbox_deltas[[l]][b, , , , drop = FALSE]
      a <- o$shape[2]; h <- o$shape[3]; w <- o$shape[4]
      o <- o$reshape(c(a, h, w))$permute(c(2, 3, 1))$reshape(-1)                 # (H, W, A)
      d <- d$reshape(c(a, 4, h, w))$permute(c(3, 4, 1, 2))$reshape(c(-1, 4))     # (H, W, A) x 4
      k <- min(pre_nms_top_n, o$numel())
      top <- o$topk(k)
      obj_l[[l]] <- top[[1]]
      box_l[[l]] <- rcnn_decode(.isel(d, top[[2]]), .isel(anchors[[l]], top[[2]]))$reshape(c(-1, 4))
      lvl_l[[l]] <- torch::torch_full(k, l, dtype = torch::torch_long(), device = device)
    }
    scores <- torch::torch_sigmoid(torch::torch_cat(obj_l))
    boxes <- clip_boxes_to_image(torch::torch_cat(box_l), image_size)
    lvl <- torch::torch_cat(lvl_l)

    keep <- remove_small_boxes(boxes, min_size)
    boxes <- .isel(boxes, keep); scores <- .isel(scores, keep); lvl <- .isel(lvl, keep)
    keep <- torch::torch_where(scores >= score_thresh)[[1]]
    boxes <- .isel(boxes, keep); scores <- .isel(scores, keep); lvl <- .isel(lvl, keep)

    keep <- batched_nms(boxes, scores, lvl, nms_thresh)
    if (keep$shape[1] > post_nms_top_n) keep <- keep[1:post_nms_top_n]
    list(boxes = .isel(boxes, keep), scores = .isel(scores, keep))
  })
}

# RoIAlign (torchvision semantics, aligned = FALSE, fixed sampling_ratio) of the
# rois of one image on one feature map [C, H, W]. boxes in image coordinates.
rcnn_roi_align_single <- function(feature, boxes, output_size, spatial_scale,
                                  sampling_ratio = 2L, chunk = 128L) {
  if (sampling_ratio <= 0) value_error("Only a positive sampling_ratio is supported.")
  C <- feature$shape[1]; H <- feature$shape[2]; W <- feature$shape[3]
  P <- output_size[1]; Q <- output_size[2]; sr <- sampling_ratio
  n <- boxes$shape[1]
  if (n == 0) return(torch_empty(c(0, C, P, Q), device = feature$device, dtype = feature$dtype))
  flat <- feature$reshape(c(C, H * W))
  dev <- feature$device
  # sample positions as in torchvision: start + ph * bin + (iy + 0.5) * bin / sr
  ph <- torch::torch_arange(0, P - 1, device = dev, dtype = feature$dtype)
  pw <- torch::torch_arange(0, Q - 1, device = dev, dtype = feature$dtype)
  iyy <- torch::torch_arange(0, sr - 1, device = dev, dtype = feature$dtype) + 0.5
  out <- vector("list", ceiling(n / chunk))
  for (ci in seq_along(out)) {
    rng <- ((ci - 1) * chunk + 1):min(ci * chunk, n)
    bx <- boxes[rng, , drop = FALSE]$to(dtype = feature$dtype)
    k <- length(rng)
    start_x <- bx[, 1] * spatial_scale; start_y <- bx[, 2] * spatial_scale
    roi_w <- torch::torch_clamp(bx[, 3] * spatial_scale - start_x, min = 1)
    roi_h <- torch::torch_clamp(bx[, 4] * spatial_scale - start_y, min = 1)
    bin_w <- roi_w / Q; bin_h <- roi_h / P
    ys <- start_y$view(c(-1, 1, 1)) + ph$view(c(1, -1, 1)) * bin_h$view(c(-1, 1, 1)) +
      iyy$view(c(1, 1, -1)) * bin_h$view(c(-1, 1, 1)) / sr                          # [k, P, sr]
    xs <- start_x$view(c(-1, 1, 1)) + pw$view(c(1, -1, 1)) * bin_w$view(c(-1, 1, 1)) +
      iyy$view(c(1, 1, -1)) * bin_w$view(c(-1, 1, 1)) / sr                          # [k, Q, sr]
    ys <- ys$view(c(k, -1, 1))$expand(c(k, P * sr, Q * sr))
    xs <- xs$view(c(k, 1, -1))$expand(c(k, P * sr, Q * sr))
    valid <- (ys >= -1) & (ys <= H) & (xs >= -1) & (xs <= W)
    y <- torch::torch_clamp(ys, min = 0, max = H - 1)
    x <- torch::torch_clamp(xs, min = 0, max = W - 1)
    y_low <- y$floor(); x_low <- x$floor()
    y_high <- torch::torch_clamp(y_low + 1, max = H - 1)
    x_high <- torch::torch_clamp(x_low + 1, max = W - 1)
    ly <- y - y_low; lx <- x - x_low; hy <- 1 - ly; hx <- 1 - lx
    w1 <- hy * hx; w2 <- hy * lx; w3 <- ly * hx; w4 <- ly * lx
    yl <- y_low$to(dtype = torch::torch_long()); yh <- y_high$to(dtype = torch::torch_long())
    xl <- x_low$to(dtype = torch::torch_long()); xh <- x_high$to(dtype = torch::torch_long())
    g <- function(yy, xx) flat[, (yy * W + xx + 1L)$reshape(-1)]$view(c(C, k, P * sr, Q * sr))
    val <- w1$unsqueeze(1) * g(yl, xl) + w2$unsqueeze(1) * g(yl, xh) +
      w3$unsqueeze(1) * g(yh, xl) + w4$unsqueeze(1) * g(yh, xh)
    val <- val * valid$unsqueeze(1)$to(dtype = val$dtype)
    val <- val$view(c(C, k, P, sr, Q, sr))$sum(dim = c(4, 6)) / (sr * sr)
    out[[ci]] <- val$permute(c(2, 1, 3, 4))
  }
  torch::torch_cat(out)$contiguous()
}

# torchvision MultiScaleRoIAlign. features: list of [B, C, H_l, W_l] levels to pool from;
# boxes: list (per image) of [N_i, 4] boxes in image coordinates.
rcnn_multiscale_roi_align <- function(features, boxes, image_size, output_size = c(7L, 7L),
                                      sampling_ratio = 2L, canonical_scale = 224, canonical_level = 4) {
  scales <- sapply(features, function(f) 2^round(log2(f$shape[3] / image_size[1])))
  lvl_min <- -log2(scales[1]); lvl_max <- -log2(scales[length(scales)])
  all_boxes <- torch::torch_cat(boxes)
  n_total <- all_boxes$shape[1]
  C <- features[[1]]$shape[2]
  res <- torch::torch_zeros(c(n_total, C, output_size[1], output_size[2]),
                            dtype = features[[1]]$dtype, device = features[[1]]$device)
  if (n_total == 0) return(res)
  if (length(features) == 1) {
    levels <- torch::torch_zeros(n_total, dtype = torch::torch_long())
  } else {
    s <- torch::torch_sqrt(box_area(all_boxes))
    levels <- torch::torch_floor(canonical_level + torch::torch_log2(s / canonical_scale) +
                                   torch_tensor(1e-6, dtype = s$dtype))
    levels <- torch::torch_clamp(levels, min = lvl_min, max = lvl_max)$to(dtype = torch::torch_long()) - lvl_min
  }
  levels <- as.integer(levels$cpu())
  img_idx <- rep(seq_along(boxes), sapply(boxes, function(b) b$shape[1]))
  for (l in seq_along(features)) {
    for (b in unique(img_idx)) {
      sel <- which(levels == (l - 1) & img_idx == b)
      if (length(sel) == 0) next
      sel_t <- torch_tensor(sel, dtype = torch::torch_long(), device = all_boxes$device)
      res[sel_t, , , ] <- rcnn_roi_align_single(features[[l]][b, , , , drop = FALSE]$reshape(features[[l]]$shape[2:4]), .isel(all_boxes, sel_t),
                                                output_size, scales[l], sampling_ratio)
    }
  }
  res
}

#' Postprocess Detections
#'
#' Processes class logits and box regression to produce final detections
#' (torchvision RoIHeads.postprocess_detections).
#'
#' @param class_logits Tensor of shape \[N, num_classes_with_bg\] - raw classification scores including background
#' @param box_regression Tensor of shape \[N, num_classes_with_bg * 4\] - box deltas for each class
#' @param proposals Tensor of shape \[N, 4\] - proposal boxes in (x1, y1, x2, y2) format
#' @param image_size Integer vector of length 2: \[height, width\]
#' @param num_classes Integer - number of classes excluding background (e.g., 90 for COCO)
#' @param score_thresh Numeric - minimum score threshold for detections
#' @param nms_thresh Numeric - NMS IoU threshold
#' @param detections_per_img Integer - maximum detections to return
#'
#' @details
#' Box regression uses standard Faster R-CNN decoding with weights (10, 10, 5, 5).
#' Background class (index 1 in class_logits) is removed, returning labels
#' in range \[1, num_classes\] which correspond to COCO classes \[1=person, 2=bicycle, ..., 90=toothbrush\].
#'
#' @return List with boxes, labels, scores tensors
#' @noRd
postprocess_detections <- function(class_logits, box_regression, proposals,
                                   image_size, num_classes, score_thresh,
                                   nms_thresh, detections_per_img) {
  device <- class_logits$device
  num_proposals <- proposals$shape[1]
  num_classes_with_bg <- num_classes + 1L

  pred_boxes <- rcnn_decode(box_regression, proposals, weights = c(10, 10, 5, 5))  # [N, K, 4]
  pred_scores <- torch::nnf_softmax(class_logits, dim = -1)

  pred_boxes <- clip_boxes_to_image(pred_boxes$reshape(c(-1, 4)), image_size)$reshape(c(num_proposals, -1, 4))
  labels <- torch_arange(0L, num_classes, device = device, dtype = torch_long())
  labels <- labels$view(c(1, -1))$expand_as(pred_scores)

  # remove background class (column 1); labels are 1..num_classes
  boxes <- pred_boxes[, 2:num_classes_with_bg, , drop = FALSE]$reshape(c(-1, 4))
  scores <- pred_scores[, 2:num_classes_with_bg, drop = FALSE]$reshape(-1)
  labels <- labels[, 2:num_classes_with_bg, drop = FALSE]$reshape(-1)

  keep <- torch::torch_where(scores > score_thresh)[[1]]
  boxes <- .isel(boxes, keep); scores <- .isel(scores, keep); labels <- .isel(labels, keep)

  keep <- remove_small_boxes(boxes, min_size = 1e-2)
  boxes <- .isel(boxes, keep); scores <- .isel(scores, keep); labels <- .isel(labels, keep)

  keep <- batched_nms(boxes, scores, labels, nms_thresh)
  if (keep$shape[1] > detections_per_img) keep <- keep[1:detections_per_img]

  list(boxes = .isel(boxes, keep), labels = .isel(labels, keep), scores = .isel(scores, keep))
}

# torchvision paste_masks_in_image: masks [N, M, M] probabilities, boxes [N, 4] -> [N, H, W]
rcnn_paste_masks <- function(masks, boxes, image_size, padding = 1L) {
  im_h <- image_size[1]; im_w <- image_size[2]
  n <- masks$shape[1]
  if (n == 0) return(torch::torch_zeros(c(0, im_h, im_w), dtype = masks$dtype, device = masks$device))
  M <- masks$shape[3]
  scale <- (M + 2 * padding) / M
  padded <- torch::nnf_pad(masks, c(padding, padding, padding, padding))
  w_half <- (boxes[, 3] - boxes[, 1]) * 0.5
  h_half <- (boxes[, 4] - boxes[, 2]) * 0.5
  x_c <- (boxes[, 3] + boxes[, 1]) * 0.5
  y_c <- (boxes[, 4] + boxes[, 2]) * 0.5
  w_half <- w_half * scale; h_half <- h_half * scale
  bx <- torch_stack(list(x_c - w_half, y_c - h_half, x_c + w_half, y_c + h_half), dim = 2)
  bx <- as.matrix(as.array(bx$to(dtype = torch::torch_long())$cpu()))
  res <- torch::torch_zeros(c(n, im_h, im_w), dtype = masks$dtype, device = masks$device)
  for (i in seq_len(n)) {
    b <- bx[i, ]
    w <- max(b[3] - b[1] + 1, 1); h <- max(b[4] - b[2] + 1, 1)
    m <- torch::nnf_interpolate(padded[i, , , drop = FALSE]$unsqueeze(1), size = c(h, w),
                                mode = "bilinear", align_corners = FALSE)[1, 1, , ]
    x0 <- max(b[1], 0); x1 <- min(b[3] + 1, im_w); y0 <- max(b[2], 0); y1 <- min(b[4] + 1, im_h)
    if (x1 > x0 && y1 > y0) {
      res[i, (y0 + 1):y1, (x0 + 1):x1] <- m[(y0 - b[2] + 1):(y1 - b[2]), (x0 - b[1] + 1):(x1 - b[1])]
    }
  }
  res
}

# Default RPN / anchor configuration of the ResNet-50 FPN models.
rcnn_mobilenet_rpn_config <- function(top_n = 1000) {
  # torchvision: anchor sizes (32, ..., 512) on each of the 3 levels ("0", "1", "pool"),
  # rpn_score_thresh = 0.05; the 320 variant uses 150 pre/post NMS proposals.
  list(anchor_sizes = rep(list(c(32, 64, 128, 256, 512)), 3),
       aspect_ratios = rep(list(c(0.5, 1, 2)), 3),
       pre_nms_top_n = top_n, post_nms_top_n = top_n,
       nms_thresh = 0.7, score_thresh = 0.05, min_size = 1e-3)
}

rcnn_resnet_rpn_config <- function() {
  list(anchor_sizes = list(32, 64, 128, 256, 512),
       aspect_ratios = rep(list(c(0.5, 1, 2)), 5),
       pre_nms_top_n = 1000, post_nms_top_n = 1000,
       nms_thresh = 0.7, score_thresh = 0, min_size = 1e-3)
}

# Generalized R-CNN inference forward shared by the Faster / Mask R-CNN models.
rcnn_forward <- function(self, images) {
  features <- self$backbone(images)
  image_size <- as.integer(images$shape[3:4])
  proposals <- rcnn_proposals(self, features, image_size)
  detections <- rcnn_detect(self, features, lapply(proposals, `[[`, "boxes"), image_size)
  if (!is.null(self$mask_head)) detections <- rcnn_masks(self, features, detections, image_size)
  list(features = features, detections = detections)
}

rcnn_proposals <- function(self, features, image_size) {
  cfg <- self$rpn_config
  rpn_out <- self$rpn(unname(features))
  shapes <- lapply(features, function(f) as.integer(f$shape[3:4]))
  anchors <- rcnn_anchors(image_size, shapes, cfg$anchor_sizes, cfg$aspect_ratios,
                          device = features[[1]]$device)
  rcnn_filter_proposals(rpn_out$objectness, rpn_out$bbox_deltas, anchors, image_size,
                        pre_nms_top_n = cfg$pre_nms_top_n, post_nms_top_n = cfg$post_nms_top_n,
                        nms_thresh = cfg$nms_thresh, score_thresh = cfg$score_thresh,
                        min_size = cfg$min_size)
}

# levels used for RoI pooling (torchvision featmap_names = c("0", "1", "2", "3"))
rcnn_pool_levels <- function(features) {
  f <- features[names(features) != "pool"]
  f[seq_len(min(4L, length(f)))]
}

rcnn_detect <- function(self, features, proposals, image_size) {
  pooled <- rcnn_multiscale_roi_align(rcnn_pool_levels(features), proposals, image_size, c(7L, 7L), 2L)
  out <- self$roi_heads(pooled)
  n <- sapply(proposals, function(p) p$shape[1])
  off <- c(0, cumsum(n))
  lapply(seq_along(proposals), function(b) {
    if (n[b] == 0) {
      return(list(boxes = torch_empty(c(0, 4)), labels = torch_empty(c(0), dtype = torch::torch_long()),
                  scores = torch_empty(c(0))))
    }
    idx <- (off[b] + 1):off[b + 1]
    postprocess_detections(out$scores[idx, , drop = FALSE], out$boxes[idx, , drop = FALSE],
                           proposals[[b]], image_size, self$num_classes, self$score_thresh,
                           self$nms_thresh, self$detections_per_img)
  })
}

rcnn_masks <- function(self, features, detections, image_size) {
  boxes <- lapply(detections, `[[`, "boxes")
  pooled <- rcnn_multiscale_roi_align(rcnn_pool_levels(features), boxes, image_size, c(14L, 14L), 2L)
  logits <- self$mask_head(pooled)
  if (!is.null(self$mask_predictor)) logits <- self$mask_predictor(logits)
  n <- sapply(boxes, function(b) b$shape[1])
  off <- c(0, cumsum(n))
  lapply(seq_along(detections), function(b) {
    d <- detections[[b]]
    if (n[b] == 0) {
      d$masks <- torch::torch_zeros(c(0, image_size[1], image_size[2]))
      d$mask_probs <- torch_empty(c(0, 28, 28))
      return(d)
    }
    idx <- torch_tensor((off[b] + 1):off[b + 1], dtype = torch::torch_long())
    lg <- .isel(logits, idx)
    ch <- (d$labels + 1L)$view(c(-1, 1, 1, 1))$expand(c(n[b], 1, lg$shape[3], lg$shape[4]))
    probs <- torch::torch_gather(lg, 2, ch)[, 1, , ]$sigmoid()
    if (probs$dim() == 2) probs <- probs$unsqueeze(1)
    d$masks <- rcnn_paste_masks(probs, d$boxes, image_size)
    d$mask_probs <- probs
    d
  })
}

# Strict state dict loading (num_batches_tracked buffers excepted).
.load_rcnn_state_dict <- function(model, state_dict) {
  model_state <- model$state_dict()
  # `num_batches_tracked` is unused at inference and stored as a 0-d tensor by PyTorch
  # (shape 1 in R) or absent (FrozenBatchNorm2d): always keep the model's own buffer.
  state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]
  for (n in grep("num_batches_tracked$", names(model_state), value = TRUE)) state_dict[[n]] <- model_state[[n]]
  missing <- setdiff(names(model_state), names(state_dict))
  unexpected <- setdiff(names(state_dict), names(model_state))
  shape <- Filter(function(n) !identical(as.integer(state_dict[[n]]$shape), as.integer(model_state[[n]]$shape)),
                  intersect(names(state_dict), names(model_state)))
  if (length(missing) || length(unexpected) || length(shape)) {
    cli_abort(c("Pretrained weights do not match the model.",
                "i" = "missing: {.val {missing}}", "i" = "unexpected: {.val {unexpected}}",
                "i" = "shape mismatch: {.val {shape}}"))
  }
  model$load_state_dict(state_dict[names(model_state)], strict = TRUE)
  invisible(model)
}

roi_heads_module <-  torch::nn_module(
    "region-of-interest head",
    initialize = function(num_classes = 90) {
      # num_classes excludes background, but predictor needs background + all classes
      num_classes_with_bg <- num_classes + 1L

      # Define box_head with named layers to match expected state dict structure
      self$box_head <- torch::nn_module(
        initialize = function() {
          self$fc6 <- torch::nn_linear(256 * 7 * 7, 1024, bias = TRUE)
          self$fc7 <- torch::nn_linear(1024, 1024, bias = TRUE)
        },
        forward = function(x) {
          x <- torch::nnf_relu(self$fc6(x))
          x <- torch::nnf_relu(self$fc7(x))
          x
        }
      )()

      self$box_predictor <- torch::nn_module(
        initialize = function() {
          self$cls_score <- torch::nn_linear(1024, num_classes_with_bg, bias = TRUE)
          self$bbox_pred <- torch::nn_linear(1024, num_classes_with_bg * 4, bias = TRUE)
        },
        forward = function(x) {
          list(
            scores = self$cls_score(x),
            boxes = self$bbox_pred(x)
          )
        }
      )()
    },
    forward = function(pooled) {
      # pooled: [N, 256, 7, 7] multi-scale RoIAlign features
      x <- self$box_head(pooled$flatten(start_dim = 2))
      self$box_predictor(x)
    }
  )

roi_heads_module_v2 <-  torch::nn_module(
    "region-of-interest head v2",
    initialize = function(num_classes = 90, in_channels = 256) {
      # num_classes excludes background, but predictor needs background + all classes
      num_classes_with_bg <- num_classes + 1L

      # V2 uses FastRCNNConvFCHead: 4 conv layers + 1 FC layer
      # PyTorch: FastRCNNConvFCHead((256, 7, 7), [256, 256, 256, 256], [1024], norm_layer=BatchNorm2d)
      # torchvision FastRCNNConvFCHead: 4 x Conv2dNormActivation (conv without bias, BN, ReLU),
      # Flatten (index 4), Linear (index 5), ReLU (index 6)
      conv_block <- function(in_ch, out_ch) {
        nn_sequential(
          nn_conv2d(in_ch, out_ch, kernel_size = 3, padding = 1, bias = FALSE),
          nn_batch_norm2d(out_ch),
          nn_relu()
        )
      }
      self$box_head <- nn_sequential(
        conv_block(in_channels, 256),
        conv_block(256, 256),
        conv_block(256, 256),
        conv_block(256, 256),
        nn_flatten(start_dim = 2),
        nn_linear(256 * 7 * 7, 1024, bias = TRUE),
        nn_relu()
      )

      self$box_predictor <- torch::nn_module(
        initialize = function() {
          self$cls_score <- torch::nn_linear(1024, num_classes_with_bg, bias = TRUE)
          self$bbox_pred <- torch::nn_linear(1024, num_classes_with_bg * 4, bias = TRUE)
        },
        forward = function(x) {
          list(
            scores = self$cls_score(x),
            boxes = self$bbox_pred(x)
          )
        }
      )()
    },
    forward = function(pooled) {
      x <- self$box_head(pooled)
      self$box_predictor(x)
    }
  )


fpn_module <- torch::nn_module(
    "feature pyramid network",
    initialize = function(in_channels, out_channels) {
      self$inner_blocks <- nn_module_list(lapply(in_channels, function(c) {
        nn_sequential(torch::nn_conv2d(c, out_channels, kernel_size = 1))
      }))
      self$layer_blocks <- nn_module_list(lapply(rep(out_channels, 4), function(i) {
        nn_sequential(torch::nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
      }))
    },
    forward = function(inputs) {
      names(inputs) <- c("c2", "c3", "c4", "c5")

      last_inner <- self$inner_blocks[[4]](inputs$c5)
      results <- vector("list", 4)
      results[[4]] <- self$layer_blocks[[4]](last_inner)

      for (i in 3:1) {
        lateral <- self$inner_blocks[[i]](inputs[[i]])
        target_size <- as.integer(lateral$shape[3:4])
        upsampled <- torch::nnf_interpolate(last_inner, size = target_size, mode = "nearest")
        last_inner <- lateral + upsampled
        results[[i]] <- self$layer_blocks[[i]](last_inner)
      }

      names(results) <- c("p2", "p3", "p4", "p5")
      # torchvision LastLevelMaxPool
      results$pool <- torch::nnf_max_pool2d(results[[4]], kernel_size = 1, stride = 2, padding = 0)
      results
    }
  )


resnet_fpn_backbone <- function(pretrained = TRUE) {
  resnet <- model_resnet50(pretrained = pretrained)

  resnet_body <- torch::nn_module(
    initialize = function() {
      self$conv1 <- resnet$conv1
      self$bn1 <- resnet$bn1
      self$relu <- resnet$relu
      self$maxpool <- resnet$maxpool
      self$layer1 <- resnet$layer1
      self$layer2 <- resnet$layer2
      self$layer3 <- resnet$layer3
      self$layer4 <- resnet$layer4
    },
    forward = function(x) {
      c2 <- x %>%
        self$conv1() %>%
        self$bn1() %>%
        self$relu() %>%
        self$maxpool() %>%
        self$layer1()

      c3 <- self$layer2(c2)
      c4 <- self$layer3(c3)
      c5 <- self$layer4(c4)

      list(c2, c3, c4, c5)
    }
  )

  backbone <- torch::nn_module(
    initialize = function() {
      self$body <- resnet_body()
      self$fpn <- fpn_module(
        in_channels = c(256, 512, 1024, 2048),
        out_channels = 256
      )
    },
    forward = function(x) {
      c2_to_c5 <- self$body(x)
      self$fpn(c2_to_c5)
    }
  )

  backbone <- backbone()
  backbone$out_channels <- 256
  backbone
}


fasterrcnn_model <- torch::nn_module(
    initialize = function(backbone, num_classes,
                          score_thresh = 0.05,
                          nms_thresh = 0.5,
                          detections_per_img = 100,
                          rpn_config = rcnn_resnet_rpn_config()) {
      self$rpn_config <- rpn_config
      self$backbone <- backbone
      self$num_classes <- num_classes

      # Store configurable detection parameters
      self$score_thresh <- score_thresh
      self$nms_thresh <- nms_thresh
      self$detections_per_img <- detections_per_img

      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head(in_channels = backbone$out_channels)
        },
        forward = function(features) {
          self$head(features)
        }
      )()

      # Use the roi_heads_module instead of inline definition
      self$roi_heads <- roi_heads_module(num_classes = num_classes)
    },

    forward = function(images) {
      rcnn_forward(self, images)
    }
  )



fpn_module_v2 <- torch::nn_module(
    "feature pyramid network v2",
    initialize = function(in_channels, out_channels) {
      self$inner_blocks <- nn_module_list(lapply(in_channels, function(c) {
        nn_sequential(
          nn_conv2d(c, out_channels, kernel_size = 1, bias = FALSE),
          nn_batch_norm2d(out_channels)
        )
      }))
      self$layer_blocks <- nn_module_list(lapply(rep(out_channels, 4), function(i) {
        nn_sequential(
          nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = FALSE),
          nn_batch_norm2d(out_channels)
        )
      }))
    },
    forward = function(inputs) {
      names(inputs) <- c("c2", "c3", "c4", "c5")

      last_inner <- self$inner_blocks[[4]](inputs$c5)
      results <- vector("list", 4)
      results[[4]] <- self$layer_blocks[[4]](last_inner)

      for (i in 3:1) {
        lateral <- self$inner_blocks[[i]](inputs[[i]])
        target_size <- as.integer(lateral$shape[3:4])
        upsampled <- torch::nnf_interpolate(last_inner, size = target_size, mode = "nearest")
        last_inner <- lateral + upsampled
        results[[i]] <- self$layer_blocks[[i]](last_inner)
      }

      names(results) <- c("p2", "p3", "p4", "p5")
      # torchvision LastLevelMaxPool
      results$pool <- torch::nnf_max_pool2d(results[[4]], kernel_size = 1, stride = 2, padding = 0)
      results
    }
  )


resnet_fpn_backbone_v2 <- function(pretrained = TRUE) {
  resnet <- model_resnet50(pretrained = pretrained)

  resnet_body <- torch::nn_module(
    initialize = function() {
      self$conv1 <- resnet$conv1
      self$bn1 <- resnet$bn1
      self$relu <- resnet$relu
      self$maxpool <- resnet$maxpool
      self$layer1 <- resnet$layer1
      self$layer2 <- resnet$layer2
      self$layer3 <- resnet$layer3
      self$layer4 <- resnet$layer4
    },
    forward = function(x) {
       c2 <- x %>%
        self$conv1() %>%
        self$bn1() %>%
        self$relu() %>%
        self$maxpool() %>%
        self$layer1()

      c3 <- self$layer2(c2)
      c4 <- self$layer3(c3)
      c5 <- self$layer4(c4)

      list(c2, c3, c4, c5)
    }
  )

  backbone <- torch::nn_module(
    initialize = function() {
      self$body <- resnet_body()
      self$fpn <- fpn_module_v2(
        in_channels = c(256, 512, 1024, 2048),
        out_channels = 256
      )
    },
    forward = function(x) {
      c2_to_c5 <- self$body(x)
      self$fpn(c2_to_c5)
    }
  )

  backbone <- backbone()
  backbone$out_channels <- 256
  backbone
}


fasterrcnn_model_v2 <- torch::nn_module(
    initialize = function(backbone, num_classes,
                          score_thresh = 0.05,
                          nms_thresh = 0.5,
                          detections_per_img = 100,
                          rpn_config = rcnn_resnet_rpn_config()) {
      self$rpn_config <- rpn_config
      self$backbone <- backbone
      self$num_classes <- num_classes

      # Store configurable detection parameters
      self$score_thresh <- score_thresh
      self$nms_thresh <- nms_thresh
      self$detections_per_img <- detections_per_img

      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head_v2(in_channels = backbone$out_channels)
        },
        forward = function(features) {
          self$head(features)
        }
      )()
      self$roi_heads <- roi_heads_module_v2(num_classes = num_classes)
    },
    forward = function(images) {
      rcnn_forward(self, images)
    }
  )



fpn_module_2level <- torch::nn_module(
    initialize = function(in_channels, out_channels) {
      self$inner_blocks <- nn_module_list(lapply(in_channels, function(c) {
        nn_sequential(nn_conv2d(c, out_channels, kernel_size = 1))
      }))
      self$layer_blocks <- nn_module_list(lapply(rep(out_channels, 2), function(i) {
        nn_sequential(nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
      }))
    },
    forward = function(inputs) {
      last_inner <- self$inner_blocks[[2]](inputs[[2]])
      results <- vector("list", 2)
      results[[2]] <- self$layer_blocks[[2]](last_inner)

      lateral <- self$inner_blocks[[1]](inputs[[1]])
      upsampled <- nnf_interpolate(last_inner, size = as.integer(lateral$shape[3:4]), mode = "nearest")
      last_inner <- lateral + upsampled
      results[[1]] <- self$layer_blocks[[1]](last_inner)

      names(results) <- c("p1", "p2")
      # torchvision LastLevelMaxPool
      results$pool <- torch::nnf_max_pool2d(results[[2]], kernel_size = 1, stride = 2, padding = 0)
      results
    }
  )



mobilenet_v3_fpn_backbone <- function(pretrained = TRUE) {
  mobilenet <- model_mobilenet_v3_large(pretrained = pretrained)

  backbone_module <- torch::nn_module(
    initialize = function() {
      self$body <- mobilenet$features
      self$fpn <- fpn_module_2level(
        in_channels = c(160, 960),
        out_channels = 256
      )
    },
    forward = function(x) {
      all_feats <- vector("list", length(self$body))

      for (i in seq_len(length(self$body))) {
        x <- self$body[[i]](x)
        all_feats[[i]] <- x
      }

      feats <- list(
        all_feats[[14]],  # 160 channels
        all_feats[[17]]   # 960 channels
      )

      self$fpn(feats)
    }
  )

  backbone <- backbone_module()
  backbone$out_channels <- 256
  backbone
}

fasterrcnn_mobilenet_model <- torch::nn_module(
    initialize = function(backbone, num_classes,
                          score_thresh = 0.05,
                          nms_thresh = 0.5,
                          detections_per_img = 100,
                          rpn_config = rcnn_mobilenet_rpn_config()) {
      self$rpn_config <- rpn_config
      self$backbone <- backbone
      self$num_classes <- num_classes

      # Store configurable detection parameters
      self$score_thresh <- score_thresh
      self$nms_thresh <- nms_thresh
      self$detections_per_img <- detections_per_img

      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head_mobilenet(in_channels = backbone$out_channels)
        },
        forward = function(features) {
          self$head(features)
        }
      )()
      self$roi_heads <- roi_heads_module(num_classes = num_classes)
    },
    forward = function(images) {
      rcnn_forward(self, images)
    }
  )




mobilenet_v3_320_fpn_backbone <- function(pretrained = TRUE) {
  mobilenet <- model_mobilenet_v3_large(pretrained = pretrained)

  backbone_module <- torch::nn_module(
    initialize = function() {
      self$body <- mobilenet$features
      self$fpn <- fpn_module_2level(
        in_channels = c(160, 960),  # output channels of layer 13 and 16
        out_channels = 256
      )
    },
    forward = function(x) {
      all_feats <- vector("list", length(self$body))

      for (i in seq_len(length(self$body))) {
        x <- self$body[[i]](x)
        all_feats[[i]] <- x
      }

      feats <- list(
        all_feats[[14]],  # 160 channels
        all_feats[[17]]   # 960 channels
      )

      self$fpn(feats)
    }
  )

  backbone <- backbone_module()
  backbone$out_channels <- 256
  backbone
}


#' Faster R-CNN Models
#'
#' Construct Faster R-CNN model variants for object-detection task.
#'
#' @param pretrained Logical. If TRUE, loads pretrained weights from local file.
#' @param progress Logical. Show progress bar during download (unused).
#' @param num_classes Number of output classes excluding background (default: 90 for COCO).
#' @param score_thresh Numeric. Minimum score threshold for detections (default: 0.05).
#' @param nms_thresh Numeric. Non-Maximum Suppression (NMS) IoU threshold for removing overlapping boxes (default: 0.5).
#' @param detections_per_img Integer. Maximum number of detections per image (default: 100).
#' @param ... Other arguments (unused).
#' @return A `fasterrcnn_model` nn_module.
#'
#' @section Task:
#' Object detection over images with bounding boxes and class labels.
#'
#' @section Input Format:
#' Input images should be `torch_tensor`s of shape
#' \verb{(batch_size, 3, H, W)} where `H` and `W` are typically around 800.
#'
#' @section Available Models:
#' \itemize{
#' \item `model_fasterrcnn_resnet50_fpn()`
#' \item `model_fasterrcnn_resnet50_fpn_v2()`
#' \item `model_fasterrcnn_mobilenet_v3_large_fpn()`
#' \item `model_fasterrcnn_mobilenet_v3_large_320_fpn()`
#' }
#'
#' @examples
#' \dontrun{
#' library(magrittr)
#' # ImageNet normalization constants, see https://pytorch.org/vision/stable/models.html
#' norm_mean <- c(0.485, 0.456, 0.406)
#' norm_std  <- c(0.229, 0.224, 0.225)
#' # Use a publicly available image of an animal
#' url <- paste0("https://upload.wikimedia.org/wikipedia/commons/thumb/",
#'        "e/ea/Morsan_Normande_vache.jpg/120px-Morsan_Normande_vache.jpg")
#' image <- magick_loader(url) %>%
#'   transform_to_tensor() %>%
#'   transform_resize(c(520, 520))
#' # ResNet backbone requires image normalization
#' input <- image  %>%
#'   transform_normalize(norm_mean, norm_std)
#' batch_normalized <- input$unsqueeze(1)    # Add batch dimension (1, 3, H, W)
#'
#' # ResNet-50 FPN V2
#' model <- model_fasterrcnn_resnet50_fpn_v2(pretrained = TRUE, detections_per_img = 5 )
#' model$eval()
#' torch::with_no_grad({pred <- model(batch_normalized)$detections[[1]]})
#' labels <- coco_classes(as.integer(pred$labels))
#'
#' # Visualize boxes
#' labels <- coco_classes(as.integer(pred$labels))
#' boxed <- draw_bounding_boxes(image, pred$boxes, labels = labels)
#' tensor_image_browse(boxed)
#'
#' # MobileNet V3 Large 320 FPN
#' batch <- image$unsqueeze(1)    # Add batch dimension (1, 3, H, W)
#' model <- model_fasterrcnn_mobilenet_v3_large_320_fpn(
#'   pretrained = TRUE, score_thresh = 0.02, nms_thresh = 0.8, detections_per_img = 5
#' )
#' model$eval()
#' torch::with_no_grad({pred <- model(batch)$detections[[1]]})
#'
#' # Visualize boxes
#' labels <- coco_classes(as.integer(pred$labels))
#' boxed <- draw_bounding_boxes(image, pred$boxes, labels = labels)
#' tensor_image_browse(boxed)
#' }
#'
#' @family object_detection_model
#' @name model_fasterrcnn
#' @rdname model_fasterrcnn
NULL

rpn_model_urls <- list(
  fasterrcnn_resnet50 = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/fasterrcnn_resnet50.pth",
    "8c519bb0e3a1a4fd94fb7bd21d51c135", "160 MB"),
  fasterrcnn_resnet50_v2 = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/fasterrcnn_resnet50_v2.pth",
    "88b414aecf00367413650dc732aa0aba", "170 MB"),
  fasterrcnn_mobilenet_v3_large = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/fasterrcnn_mobilenet_v3_large.pth",
    "58eba0ba379ed1497da8aa1adb8b7a7e", "75 MB"),
  fasterrcnn_mobilenet_v3_large_320 = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/fasterrcnn_mobilenet_v3_large_320.pth",
    "bd711fc4bb0da7fce38ca9916fc98753", "75 MB")
)


#' @describeIn model_fasterrcnn Faster R-CNN with ResNet-50 FPN
#' @export
model_fasterrcnn_resnet50_fpn <- function(pretrained = FALSE, progress = TRUE,
                                          num_classes = 90,
                                          score_thresh = 0.05,
                                          nms_thresh = 0.5,
                                          detections_per_img = 100,
                                          ...) {
  backbone <- resnet_fpn_backbone(pretrained = pretrained)
  model <- fasterrcnn_model(backbone, num_classes = num_classes,
                            score_thresh = score_thresh,
                            nms_thresh = nms_thresh,
                            detections_per_img = detections_per_img)
  if (pretrained && num_classes != 90)
    cli_abort("Pretrained weights require num_classes = 90 (excluding background).")

  if (pretrained) {
    r <- rpn_model_urls$fasterrcnn_resnet50
    name <- "fasterrcnn_resnet50"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "fasterrcnn")
    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)
    .load_rcnn_state_dict(model, .rename_fasterrcnn_state_dict(state_dict))
  }

  model
}


#' @describeIn model_fasterrcnn Faster R-CNN with ResNet-50 FPN V2
#' @export
model_fasterrcnn_resnet50_fpn_v2 <- function(pretrained = FALSE, progress = TRUE,
                                             num_classes = 90,
                                             score_thresh = 0.05,
                                             nms_thresh = 0.5,
                                             detections_per_img = 100,
                                             ...) {
  backbone <- resnet_fpn_backbone_v2(pretrained = pretrained)
  model <- fasterrcnn_model_v2(backbone, num_classes = num_classes,
                               score_thresh = score_thresh,
                               nms_thresh = nms_thresh,
                               detections_per_img = detections_per_img)

  if (pretrained && num_classes != 90)
    cli_abort("Pretrained weights require num_classes = 90 (excluding background).")

  if (pretrained) {
    r <- rpn_model_urls$fasterrcnn_resnet50_v2
    name <- "fasterrcnn_resnet50_v2"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "fasterrcnn")
    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }
    state_dict <- torch::load_state_dict(state_dict_path)

    .load_rcnn_state_dict(model, state_dict)
  }

  model
}


#' @describeIn model_fasterrcnn Faster R-CNN with MobileNet V3 Large FPN
#' @export
model_fasterrcnn_mobilenet_v3_large_fpn <- function(pretrained = FALSE,
                                                    progress = TRUE,
                                                    num_classes = 90,
                                                    score_thresh = 0.05,
                                                    nms_thresh = 0.5,
                                                    detections_per_img = 100,
                                                    ...) {
  backbone <- mobilenet_v3_fpn_backbone(pretrained = pretrained)
  model <- fasterrcnn_mobilenet_model(backbone, num_classes = num_classes,
                                      score_thresh = score_thresh,
                                      nms_thresh = nms_thresh,
                                      detections_per_img = detections_per_img,
                                      rpn_config = rcnn_mobilenet_rpn_config(1000))

  if (pretrained && num_classes != 90)
    cli_abort("Pretrained weights require num_classes = 90 (excluding background).")

  if (pretrained) {
    r <- rpn_model_urls$fasterrcnn_mobilenet_v3_large
    name <- "fasterrcnn_mobilenet_v3_large"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "fasterrcnn")
    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)
    .load_rcnn_state_dict(model, .rename_fasterrcnn_large_state_dict(state_dict))
  }

  model
}


#' @describeIn model_fasterrcnn Faster R-CNN with MobileNet V3 Large 320 FPN
#' @export
model_fasterrcnn_mobilenet_v3_large_320_fpn <- function(pretrained = FALSE,
                                                        progress = TRUE,
                                                        num_classes = 90,
                                                        score_thresh = 0.05,
                                                        nms_thresh = 0.5,
                                                        detections_per_img = 100,
                                                        ...) {
  backbone <- mobilenet_v3_320_fpn_backbone(pretrained = pretrained)
  model <- fasterrcnn_mobilenet_model(backbone, num_classes = num_classes,
                                      score_thresh = score_thresh,
                                      nms_thresh = nms_thresh,
                                      detections_per_img = detections_per_img,
                                      rpn_config = rcnn_mobilenet_rpn_config(150))

  if (pretrained && num_classes != 90)
    cli_abort("Pretrained weights require num_classes = 90 (excluding background).")

  if (pretrained) {
    r <- rpn_model_urls$fasterrcnn_mobilenet_v3_large_320
    name <- "fasterrcnn_mobilenet_v3_large_320"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "fasterrcnn")
    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)
    .load_rcnn_state_dict(model, .rename_fasterrcnn_large_state_dict(state_dict))
  }

  model
}

#' @importFrom stats setNames
.rename_fasterrcnn_state_dict <- function(state_dict) {
  . <- NULL # Nulling strategy for no visible binding check Note
  new_names <- names(state_dict) %>%
    # add ".0" to inner_blocks + layer_blocks layer renaming
    sub(pattern = "(inner_blocks\\.[0-3]\\.)", replacement = "\\10\\.", x = .) %>%
    sub(pattern = "(layer_blocks\\.[0-3]\\.)", replacement = "\\10\\.", x = .) %>%
    # add ".0.0" to rpn.head.conv
    sub(pattern = "(rpn\\.head\\.conv\\.)", replacement = "\\10\\.0\\.", x = .)

  # Recreate a list with renamed keys
  setNames(state_dict[names(state_dict)], new_names)
}


.rename_fasterrcnn_large_state_dict <- function(state_dict) {
  . <- NULL # Nulling strategy for no visible binding check Note
  new_names <- names(.rename_fasterrcnn_state_dict(state_dict)) %>%
    # turn bn into 'O' value and conv into '1' value
    sub(pattern = "(block\\.[0-3]\\.)0\\.", replacement = "\\1conv\\.", x = .) %>%
    sub(pattern = "(block\\.[0-3]\\.)1\\.", replacement = "\\1bn\\.", x = .) %>%
    sub(pattern = "(body\\.0\\.)0\\.", replacement = "\\1conv\\.", x = .) %>%
    sub(pattern = "(body\\.0\\.)1\\.", replacement = "\\1bn\\.", x = .) %>%
    sub(pattern = "(body\\.16\\.)0\\.", replacement = "\\1conv\\.", x = .) %>%
    sub(pattern = "(body\\.16\\.)1\\.", replacement = "\\1bn\\.", x = .)

  # Recreate a list with renamed keys
  setNames(state_dict[names(state_dict)], new_names)
}
