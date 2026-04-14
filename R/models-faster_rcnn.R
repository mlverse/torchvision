rpn_head <- torch::nn_module(
    "rpn_head",
    initialize = function(in_channels, num_anchors = 3) {
      self$conv <- nn_sequential(nn_sequential(nn_conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)))
      self$cls_logits <- nn_conv2d(in_channels, num_anchors, kernel_size = 1)
      self$bbox_pred <- nn_conv2d(in_channels, num_anchors * 4, kernel_size = 1)
    },
    forward = function(features) {
      objectness_list <- list()
      bbox_reg_list <- list()

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
      # The pretrained checkpoint stacks two Conv2d → ReLU
      # blocks with bias enabled. Mirror that layout so
      # the parameter names and shapes line up with the weight file.
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
      objectness_list <- list()
      bbox_reg_list <- list()

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
      objectness <- list()
      bbox_deltas <- list()

      for (x in features) {
        t <- nnf_relu(self$conv(x))
        objectness[[length(objectness) + 1]] <- self$cls_logits(t)
        bbox_deltas[[length(bbox_deltas) + 1]] <- self$bbox_pred(t)
      }

      list(objectness = objectness, bbox_deltas = bbox_deltas)
    }
  )


#' @importFrom torch torch_meshgrid torch_stack torch_tensor torch_stack torch_zeros_like torch_max torch_float32
#
# generate_level_anchors
#
# Builds all anchor boxes for one FPN level (pure R + torch).
#
# The pretrained Faster R-CNN weights were trained with:
#   sizes  = 32, 64, 128, 256  (one per level, 4 levels)
#   ratios = 0.5, 1.0, 2.0    (3 anchors per grid location)
#
# h, w       : feature map height and width
# stride     : stride of this FPN level = image_size / feature_map_size
# anchor_size: the single anchor size for this level (e.g. 64 for p3)
# ratios     : aspect ratios, default c(0.5, 1.0, 2.0)
#
# Returns a CPU tensor of shape [H * W * 3, 4] in (x1, y1, x2, y2) image coords.
generate_level_anchors <- function(h, w, stride,
                                   anchor_size,
                                   ratios = c(0.5, 1.0, 2.0)) {
  # ---- base anchors at origin (plain R arithmetic) ----------------------
  area <- anchor_size^2
  ws   <- sqrt(area / ratios)    # width  for each ratio  [length 3]
  hs   <- ws * ratios            # height for each ratio  [length 3]

  # Each base anchor is (x1, y1, x2, y2) centered at (0,0)
  # shape: [3, 4] as a plain R matrix
  base <- cbind(-ws / 2, -hs / 2, ws / 2, hs / 2)   # [3, 4]
  base_t <- torch_tensor(base, dtype = torch_float32())  # [3, 4]

  # ---- grid of cell centers in image coordinates ------------------------
  # In R, torch_arange(a, b) is INCLUSIVE of b, so use w-1 to get exactly w values
  cx <- (torch_arange(0L, as.integer(w) - 1L, dtype = torch_float32()) + 0.5) * stride  # [W]
  cy <- (torch_arange(0L, as.integer(h) - 1L, dtype = torch_float32()) + 0.5) * stride  # [H]


  g  <- torch_meshgrid(list(cy, cx), indexing = "ij")  # each [H, W]
  # Flatten and stack into shifts [H*W, 4]: (cx, cy, cx, cy)
  cy_flat <- g[[1]]$flatten()   # [H*W]
  cx_flat <- g[[2]]$flatten()   # [H*W]
  # unsqueeze to [N,1] then cat along dim 2 -> [N,4]
  shifts  <- torch_cat(list(cx_flat$unsqueeze(2),
                            cy_flat$unsqueeze(2),
                            cx_flat$unsqueeze(2),
                            cy_flat$unsqueeze(2)), dim = 2)  # [H*W, 4]


  # ---- broadcast: [H*W, 1, 4] + [1, 3, 4] = [H*W, 3, 4] ---------------
  all_anchors <- shifts$unsqueeze(2) + base_t$unsqueeze(1)

  # flatten to [H*W*3, 4]
  all_anchors$reshape(c(-1L, 4L))
}


# decode_boxes: apply predicted (dx,dy,dw,dh) deltas to reference anchor boxes
# anchors: [N, 4]  (x1, y1, x2, y2)
# deltas:  [N, 4]  (dx, dy, dw, dh)
# returns: [N, 4]  decoded boxes in (x1, y1, x2, y2) format
decode_boxes <- function(anchors, deltas) {
  widths  <- anchors[, 3] - anchors[, 1]   # anchor widths
  heights <- anchors[, 4] - anchors[, 2]   # anchor heights
  ctr_x   <- anchors[, 1] + widths  / 2   # anchor center x
  ctr_y   <- anchors[, 2] + heights / 2   # anchor center y

  dx <- deltas[, 1]
  dy <- deltas[, 2]
  dw <- torch::torch_clamp(deltas[, 3], max = 4.135)  # clamp exp to avoid overflow
  dh <- torch::torch_clamp(deltas[, 4], max = 4.135)

  pred_ctr_x <- ctr_x + dx * widths
  pred_ctr_y <- ctr_y + dy * heights
  pred_w     <- torch::torch_exp(dw) * widths
  pred_h     <- torch::torch_exp(dh) * heights

  x1 <- pred_ctr_x - pred_w / 2
  y1 <- pred_ctr_y - pred_h / 2
  x2 <- pred_ctr_x + pred_w / 2
  y2 <- pred_ctr_y + pred_h / 2

  # Bug fix: was dim=2 which gave [N,1,4]; -1 (last dim) gives correct [N,4]
  torch::torch_stack(list(x1, y1, x2, y2), dim = -1)
}

#' @importFrom torch nnf_grid_sample torch_empty
#
# generate_proposals
#
# Runs the RPN: for each FPN level, generate anchors, decode predicted deltas,
# score by objectness, then filter + NMS to get final region proposals.
#
# anchor_sizes: one size per FPN level (e.g. 32, 64, 128, 256 for 4 levels)
#               must match what the pretrained RPN head was trained with.
generate_proposals <- function(features, rpn_out, image_size, anchor_sizes, batch_idx,
                               score_thresh = 0.0, nms_thresh = 0.7,
                               pre_nms_top_n = 1000L, post_nms_top_n = 1000L) {
  device       <- rpn_out$objectness[[1]]$device
  all_proposals <- torch_empty(0L, 4L, device = device)
  all_scores    <- torch_empty(0L,     device = device)

  for (i in seq_along(features)) {
    objectness <- rpn_out$objectness[[i]][batch_idx, , , ]  # [A, H, W]
    deltas     <- rpn_out$bbox_deltas[[i]][batch_idx, , , ] # [A*4, H, W]

    c(a, h, w) %<-% objectness$shape

    # stride of this FPN level = image_width / feature_map_width
    stride <- as.integer(image_size[2]) / w

    # generate anchors: [H*W*3, 4]  (3 aspect ratios per location)
    anchors <- generate_level_anchors(h, w, stride,
                                      anchor_size = anchor_sizes[[i]],
                                      ratios = c(0.5, 1.0, 2.0))
    anchors <- anchors$to(device = device)

    objectness <- objectness$sigmoid()$flatten()            # [H*W*3]
    deltas     <- deltas$permute(c(2, 3, 1))$reshape(c(-1, 4))  # [H*W*3, 4]

    proposals <- decode_boxes(anchors, deltas)
    proposals <- clip_boxes_to_image(proposals, image_size)

    all_proposals <- torch::torch_cat(list(all_proposals, proposals), dim = 1L)
    all_scores    <- torch::torch_cat(list(all_scores, objectness),   dim = 1L)
  }

  # keep high-confidence proposals and suppress overlapping ones
  scores    <- all_scores$flatten()
  keep      <- scores > score_thresh
  proposals <- all_proposals[keep, ]
  scores    <- scores[keep]

  if (proposals$shape[1] > pre_nms_top_n) {
    topk      <- torch::torch_topk(scores, pre_nms_top_n)
    keep_idx  <- topk[[2]]
    proposals <- proposals[keep_idx, ]
    scores    <- scores[keep_idx]
  }

  if (proposals$shape[1] > 0) {
    keep_idx  <- nms(proposals, scores, nms_thresh)
    proposals <- proposals[keep_idx, ]
    scores    <- scores[keep_idx]
    
    if (proposals$shape[1] > post_nms_top_n) {
      proposals <- proposals[1:post_nms_top_n, ]
      scores    <- scores[1:post_nms_top_n]
    }
  } else {
    proposals <- torch_empty(c(0, 4), device = device, dtype = torch_float32())
  }

  list(proposals = proposals)
}

# roi_align_single_level
#
# Pools features for a set of proposals from ONE feature map level.
#
# feature_map  : [1, C, H_feat, W_feat]  — a single FPN level (already batch-selected)
# proposals    : [N, 4]  in image coordinates (x1, y1, x2, y2)
# spatial_scale: image_size / feature_map_size  (e.g. 1/4 for stride-4 level)
# output_size  : e.g. c(7L, 7L)
#
# Returns [N, C, output_size[1], output_size[2]]
roi_align_single_level <- function(feature_map, proposals,
                                   spatial_scale = 1/4,
                                   output_size   = c(7L, 7L)) {
  num_rois <- proposals$size(1)
  if (num_rois == 0) {
    return(torch_empty(c(0, feature_map$size(2),
                         output_size[1], output_size[2]),
                       device = feature_map$device))
  }

  channels <- feature_map$size(2)
  h_feat   <- feature_map$size(3)
  w_feat   <- feature_map$size(4)

  # Step 1: scale proposals from image space → feature-map space
  boxes_fm <- proposals$to(dtype = torch_float()) * spatial_scale  # [N, 4]

  # Step 2: normalize feature-map coordinates to [-1, 1] for grid_sample
  # Using align_corners = FALSE convention: pixel i covers [i, i+1) in feature space,
  # so the center of the whole map spans [-1, 1] with formula: coord / size * 2 - 1
  x1 <- boxes_fm[, 1] / w_feat * 2 - 1
  y1 <- boxes_fm[, 2] / h_feat * 2 - 1
  x2 <- boxes_fm[, 3] / w_feat * 2 - 1
  y2 <- boxes_fm[, 4] / h_feat * 2 - 1

  # Step 3: build a [N, out_h, out_w, 2] sampling grid
  # Each ROI gets its own evenly-spaced grid between its (x1,y1) and (x2,y2)
  rel   <- torch_linspace(0, 1, output_size[1], device = feature_map$device)  # [M]
  grids <- torch_meshgrid(list(rel, rel), indexing = "ij")  # each [M, M]
  rel_y <- grids[[1]]  # [out_h, out_w]
  rel_x <- grids[[2]]

  sampling_x <- x1$view(c(-1, 1, 1)) + rel_x$view(c(1, output_size[1], output_size[2])) * (x2 - x1)$view(c(-1, 1, 1))
  sampling_y <- y1$view(c(-1, 1, 1)) + rel_y$view(c(1, output_size[1], output_size[2])) * (y2 - y1)$view(c(-1, 1, 1))

  grid <- torch_stack(list(sampling_x, sampling_y), dim = -1)  # [N, out_h, out_w, 2]

  # Step 4: bilinear sample from the feature map
  # expand feature_map [1, C, H, W] -> [N, C, H, W] cheaply
  input_expanded <- feature_map$expand(c(num_rois, channels, h_feat, w_feat))

  nnf_grid_sample(
    input_expanded,
    grid,
    mode          = "bilinear",
    padding_mode  = "border",
    align_corners = FALSE
  )  # [N, C, out_h, out_w]
}


# roi_align_fpn
#
# Multi-scale ROI Align following the FPN paper (Lin et al. 2017).
# Assigns each proposal to the FPN level whose stride best matches the proposal size,
# then calls roi_align_single_level for each group.
#
# features : named list with elements "p2","p3","p4","p5"  — each [1, C, H_l, W_l]
# proposals: [N, 4]  image coordinates
# output_size: pooled spatial size, default c(7L, 7L)
#
# Returns [N, C, out_h, out_w]
roi_align_fpn <- function(features, proposals, output_size = c(7L, 7L)) {
  # Spatial scales for each FPN level (stride = 4, 8, 16, 32)
  spatial_scales <- c(1/4, 1/8, 1/16, 1/32)
  level_names    <- c("p2", "p3", "p4", "p5")

  n_boxes    <- proposals$size(1)
  c_channels <- features[[1]]$size(2)
  out        <- torch_zeros(c(n_boxes, c_channels, output_size[1], output_size[2]),
                            dtype  = features[[1]]$dtype,
                            device = features[[1]]$device)

  if (n_boxes == 0) return(out)

  # Assign each proposal to a level based on its sqrt(area)
  # Formula from FPN paper Eq.(1): level = floor(k0 + log2(sqrt(area) / 224))
  # k0=4, clamped to [2, 5] → list index [1, 4]
  ws <- proposals[, 3] - proposals[, 1]
  hs <- proposals[, 4] - proposals[, 2]
  ws[ws <= 0] <- 1
  hs[hs <= 0] <- 1
  area_sqrt   <- (ws * hs)$sqrt()
  target_lvl  <- (torch_log2(area_sqrt / 224) + 4)$floor()
  target_lvl  <- torch_clamp(target_lvl, min = 2, max = 5)  # clamp to levels 2-5
  list_idx    <- (target_lvl - 2)$to(dtype = torch_long()) + 1L  # convert to 1-based index

  for (lvl in 1:4) {
    mask <- list_idx$eq(lvl)$squeeze()
    idx  <- torch_nonzero(mask)$squeeze()
    if (idx$numel() == 0) next
    if (idx$dim() == 0) idx <- idx$unsqueeze(1)  # handle single-element

    pooled_lvl <- roi_align_single_level(
      feature_map   = features[[level_names[lvl]]],
      proposals     = proposals[idx, , drop = FALSE],
      spatial_scale = spatial_scales[lvl],
      output_size   = output_size
    )
    out[idx, , , ] <- pooled_lvl
  }

  out
}

roi_heads_module <-  torch::nn_module(
    "region-of-interest head v2",
    initialize = function(num_classes = 91) {
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
          self$cls_score <- torch::nn_linear(1024, num_classes, bias = TRUE)
          self$bbox_pred <- torch::nn_linear(1024, num_classes * 4, bias = TRUE)
        },
        forward = function(x) {
          list(
            scores = self$cls_score(x),
            boxes = self$bbox_pred(x)
          )
        }
      )()
    },
    forward = function(features, proposals, batch_idx) {
      # Select the batch item from each feature level: [1, C, H, W]
      feature_maps <- list(
        p2 = features[["p2"]][batch_idx, , , , drop = FALSE],
        p3 = features[["p3"]][batch_idx, , , , drop = FALSE],
        p4 = features[["p4"]][batch_idx, , , , drop = FALSE],
        p5 = features[["p5"]][batch_idx, , , , drop = FALSE]
      )
      # Multi-scale ROI pooling across all FPN levels
      pooled <- roi_align_fpn(feature_maps, proposals, output_size = c(7L, 7L))
      # Flatten [N, C, 7, 7] -> [N, C*7*7] then through the box head
      x <- self$box_head(pooled$flatten(start_dim = 2))
      self$box_predictor(x)
    }
  )


roi_heads_module_v2 <- torch::nn_module(
    "region-of-interest head v2",
    initialize = function(num_classes = 91) {
      # The pretrained weights expect four (Linear -> BN -> ReLU) blocks
      # followed by a final linear layer at index "4".
      block <- function() {
        nn_sequential(
          nn_linear(1024, 1024, bias = FALSE),
          nn_batch_norm1d(1024),
          nn_relu()
        )
      }

      layers <- list(
        nn_sequential(
          nn_linear(256 * 7 * 7, 1024, bias = FALSE),
          nn_batch_norm1d(1024),
          nn_relu()
        ),
        block(),
        block(),
        block(),
        nn_linear(1024, 1024, bias = TRUE)
      )
      self$box_head <- do.call(nn_sequential, layers)
      self$box_predictor <- torch::nn_module(
        initialize = function() {
          self$cls_score <- torch::nn_linear(1024, num_classes, bias = TRUE)
          self$bbox_pred <- torch::nn_linear(1024, num_classes * 4, bias = TRUE)
        },
        forward = function(x) {
          list(
            scores = self$cls_score(x),
            boxes = self$bbox_pred(x)
          )
        }
      )()
    },
    forward = function(features, proposals, batch_idx) {
      # Select the batch item from each feature level: [1, C, H, W]
      feature_maps <- list(
        p2 = features[["p2"]][batch_idx, , , , drop = FALSE],
        p3 = features[["p3"]][batch_idx, , , , drop = FALSE],
        p4 = features[["p4"]][batch_idx, , , , drop = FALSE],
        p5 = features[["p5"]][batch_idx, , , , drop = FALSE]
      )
      pooled <- roi_align_fpn(feature_maps, proposals, output_size = c(7L, 7L))
      x <- self$box_head(pooled$flatten(start_dim = 2))
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
      results <- list()
      results[[4]] <- self$layer_blocks[[4]](last_inner)

      for (i in 3:1) {
        lateral <- self$inner_blocks[[i]](inputs[[i]])
        target_size <- as.integer(lateral$shape[3:4])
        upsampled <- torch::nnf_interpolate(last_inner, size = target_size, mode = "nearest")
        last_inner <- lateral + upsampled
        results[[i]] <- self$layer_blocks[[i]](last_inner)
      }

      names(results) <- c("p2", "p3", "p4", "p5")
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
                          detections_per_img = 100) {
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
      features   <- self$backbone(images)
      rpn_out    <- self$rpn(features)
      device     <- images$device           # inherit device from input (MPS / CUDA / CPU)

      batch_size <- images$shape[1]
      image_size <- images$shape[3:4]
      final_results <- list()

      for (b in seq_len(batch_size)) {
        props <- generate_proposals(features, rpn_out, image_size,
                                    anchor_sizes = list(32, 64, 128, 256),
                                    batch_idx = b)

        if (props$proposals$shape[1] == 0) {
          empty <- list(
            boxes  = torch_empty(c(0, 4),  device = device),
            labels = torch_empty(c(0), dtype = torch::torch_long(), device = device),
            scores = torch_empty(c(0),      device = device)
          )
          final_results[[b]] <- empty
          next
        }

        detections <- self$roi_heads(features, props$proposals, batch_idx = b)

        # Softmax over all 91 classes: [N, 91]
        scores <- torch::nnf_softmax(detections$scores, dim = 2)

        # Skip background (index 1 in R = class 0 in COCO) when finding best class.
        # Slice columns 2:91 (the 90 foreground classes) then add 1 back to labels
        # so they map to the original 91-class indexing.
        fg_scores <- scores[, 2:self$num_classes]  # [N, 90] — foreground only
        max_fg    <- torch::torch_max(fg_scores, dim = 2)
        final_scores <- max_fg[[1]]                # highest foreground score
        final_labels <- max_fg[[2]] + 1L           # shift back to 1-based 91-class space

        # Pick the bounding-box deltas for the predicted class
        box_reg    <- detections$boxes$view(c(-1, self$num_classes, 4))
        gather_idx <- final_labels$unsqueeze(2)$unsqueeze(3)$expand(c(-1, 1, 4))
        final_boxes <- box_reg$gather(2, gather_idx)$squeeze(2)

        final_boxes <- decode_boxes(props$proposals, final_boxes)
        final_boxes <- clip_boxes_to_image(final_boxes, image_size)

        # Filter by score threshold
        keep <- final_scores > self$score_thresh
        num_detections <- torch::torch_sum(keep)$item()

        if (num_detections > 0) {
          final_boxes <- final_boxes[keep, ]
          final_labels <- final_labels[keep]
          final_scores <- final_scores[keep]

          # Apply NMS to remove overlapping detections
          if (final_boxes$shape[1] > 1) {
            nms_keep <- nms(final_boxes, final_scores, self$nms_thresh)
            final_boxes <- final_boxes[nms_keep, ]
            final_labels <- final_labels[nms_keep]
            final_scores <- final_scores[nms_keep]
          }

          # Limit detections per image
          n_det <- final_scores$shape[1]
          if (n_det > self$detections_per_img) {
            top_k <- torch::torch_topk(final_scores, self$detections_per_img)
            top_idx <- top_k[[2]]
            final_boxes <- final_boxes[top_idx, ]
            final_labels <- final_labels[top_idx]
            final_scores <- final_scores[top_idx]
          }
        } else {
          final_boxes  <- torch_empty(c(0, 4), device = device)
          final_labels <- torch_empty(c(0), dtype = torch::torch_long(), device = device)
          final_scores <- torch_empty(c(0),    device = device)
        }
        final_results[[b]] <- list(
          boxes = final_boxes,
          labels = final_labels,
          scores = final_scores
        )
      }
      list(features = features, detections = final_results)
    }
  )



fpn_module_v2 <- torch::nn_module(
    "feature pyramid network v2",
    initialize = function(in_channels, out_channels) {
      self$inner_blocks <- nn_module_list(lapply(in_channels, function(c) {
        nn_sequential(
          nn_conv2d(c, out_channels, kernel_size = 1, bias = FALSE),
          nn_batch_norm2d(out_channels),
          nn_relu()
        )
      }))
      self$layer_blocks <- nn_module_list(lapply(rep(out_channels, 4), function(i) {
        nn_sequential(
          nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = FALSE),
          nn_batch_norm2d(out_channels),
          nn_relu()
        )
      }))
    },
    forward = function(inputs) {
      names(inputs) <- c("c2", "c3", "c4", "c5")

      last_inner <- self$inner_blocks[[4]](inputs$c5)
      results <- list()
      results[[4]] <- self$layer_blocks[[4]](last_inner)

      for (i in 3:1) {
        lateral <- self$inner_blocks[[i]](inputs[[i]])
        target_size <- as.integer(lateral$shape[3:4])
        upsampled <- torch::nnf_interpolate(last_inner, size = target_size, mode = "nearest")
        last_inner <- lateral + upsampled
        results[[i]] <- self$layer_blocks[[i]](last_inner)
      }

      names(results) <- c("p2", "p3", "p4", "p5")
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
                          detections_per_img = 100) {
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
      features   <- self$backbone(images)
      rpn_out    <- self$rpn(features)
      device     <- images$device           # inherit device from input

      batch_size <- images$shape[1]
      image_size <- images$shape[3:4]
      final_results <- list()

      for (b in seq_len(batch_size)) {
        props <- generate_proposals(features, rpn_out, image_size,
                                    anchor_sizes = list(32, 64, 128, 256),
                                    batch_idx = b)

        if (props$proposals$shape[1] == 0) {
          empty <- list(
            boxes  = torch_empty(c(0, 4),  device = device),
            labels = torch_empty(c(0), dtype = torch::torch_long(), device = device),
            scores = torch_empty(c(0),      device = device)
          )
          final_results[[b]] <- empty
          next
        }

        detections <- self$roi_heads(features, props$proposals, batch_idx = b)

        scores    <- nnf_softmax(detections$scores, dim = 2)  # [N, 91]

        # Skip background class (column 1 = class 0) when picking the best label
        fg_scores    <- scores[, 2:self$num_classes]  # [N, 90]
        max_fg       <- torch_max(fg_scores, dim = 2)
        final_scores <- max_fg[[1]]
        final_labels <- max_fg[[2]] + 1L  # shift back to 91-class indexing

        box_reg    <- detections$boxes$view(c(-1, self$num_classes, 4))
        gather_idx <- final_labels$unsqueeze(2)$unsqueeze(3)$expand(c(-1, 1, 4))
        final_boxes <- box_reg$gather(2, gather_idx)$squeeze(2)

        final_boxes <- decode_boxes(props$proposals, final_boxes)
        final_boxes <- clip_boxes_to_image(final_boxes, image_size)

        # Filter by score threshold
        keep <- final_scores > self$score_thresh
        num_detections <- torch::torch_sum(keep)$item()

        if (num_detections > 0) {
          final_boxes <- final_boxes[keep, ]
          final_labels <- final_labels[keep]
          final_scores <- final_scores[keep]

          # Apply NMS to remove overlapping detections
          if (final_boxes$shape[1] > 1) {
            nms_keep <- nms(final_boxes, final_scores, self$nms_thresh)
            final_boxes <- final_boxes[nms_keep, ]
            final_labels <- final_labels[nms_keep]
            final_scores <- final_scores[nms_keep]
          }

          # Limit detections per image
          n_det <- final_scores$shape[1]
          if (n_det > self$detections_per_img) {
            top_k <- torch::torch_topk(final_scores, self$detections_per_img)
            top_idx <- top_k[[2]]
            final_boxes <- final_boxes[top_idx, ]
            final_labels <- final_labels[top_idx]
            final_scores <- final_scores[top_idx]
          }
        } else {
          final_boxes  <- torch_empty(c(0, 4), device = device)
          final_labels <- torch_empty(c(0), dtype = torch::torch_long(), device = device)
          final_scores <- torch_empty(c(0),    device = device)
        }
        final_results[[b]] <- list(
          boxes = final_boxes,
          labels = final_labels,
          scores = final_scores
        )
      }
      list(features = features, detections = final_results)
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
      results <- list()
      results[[2]] <- self$layer_blocks[[2]](last_inner)

      lateral <- self$inner_blocks[[1]](inputs[[1]])
      upsampled <- nnf_interpolate(last_inner, size = as.integer(lateral$shape[3:4]), mode = "nearest")
      last_inner <- lateral + upsampled
      results[[1]] <- self$layer_blocks[[1]](last_inner)

      names(results) <- c("p1", "p2")
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
      all_feats <- list()

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
                          detections_per_img = 100) {
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
      features   <- self$backbone(images)
      rpn_out    <- self$rpn(features)
      device     <- images$device           # inherit device from input

      batch_size <- images$shape[1]
      image_size <- images$shape[3:4]
      final_results <- list()

      for (b in seq_len(batch_size)) {
        props <- generate_proposals(features, rpn_out, image_size,
                                    anchor_sizes = list(64, 128),
                                    batch_idx = b,
                                    score_thresh = self$score_thresh,
                                    nms_thresh = self$nms_thresh)

        if (props$proposals$shape[1] == 0) {
          empty <- list(
            boxes  = torch_empty(c(0, 4),  device = device),
            labels = torch_empty(c(0), dtype = torch::torch_long(), device = device),
            scores = torch_empty(c(0),      device = device)
          )
          return(list(features = features, detections = list(empty)))
        }

        detections <- self$roi_heads(features, props$proposals, batch_idx = b)

        scores    <- nnf_softmax(detections$scores, dim = 2)  # [N, 91]

        fg_scores    <- scores[, 2:self$num_classes]
        max_fg       <- torch_max(fg_scores, dim = 2)
        final_scores <- max_fg[[1]]
        final_labels <- max_fg[[2]] + 1L

        box_reg    <- detections$boxes$view(c(-1, self$num_classes, 4))
        gather_idx <- final_labels$unsqueeze(2)$unsqueeze(3)$expand(c(-1, 1, 4))
        final_boxes <- box_reg$gather(2, gather_idx)$squeeze(2)

        final_boxes <- decode_boxes(props$proposals, final_boxes)
        final_boxes <- clip_boxes_to_image(final_boxes, image_size)

        # Filter by score threshold
        keep <- final_scores > self$score_thresh
        if (torch::torch_sum(keep)$item() > 0) {
          final_boxes <- final_boxes[keep, ]
          final_labels <- final_labels[keep]
          final_scores <- final_scores[keep]

          # Apply NMS to remove overlapping detections
          if (final_boxes$shape[1] > 1) {
            nms_keep <- nms(final_boxes, final_scores, self$nms_thresh)
            final_boxes <- final_boxes[nms_keep, ]
            final_labels <- final_labels[nms_keep]
            final_scores <- final_scores[nms_keep]
          }

          # Limit detections per image
          n_det <- final_scores$shape[1]
          if (n_det > self$detections_per_img) {
            top_k <- torch::torch_topk(final_scores, self$detections_per_img)
            top_idx <- top_k[[2]]
            final_boxes <- final_boxes[top_idx, ]
            final_labels <- final_labels[top_idx]
            final_scores <- final_scores[top_idx]
          }
        } else {
          final_boxes  <- torch_empty(c(0, 4), device = device)
          final_labels <- torch_empty(c(0), dtype = torch::torch_long(), device = device)
          final_scores <- torch_empty(c(0), device = device)
        }
        final_results[[b]] <- list(
          boxes = final_boxes,
          labels = final_labels,
          scores = final_scores
        )
      }
      list(features = features, detections = final_results)
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
      all_feats <- list()

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
#' @param num_classes Number of output classes (default: 91 for COCO).
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
#' norm_mean <- c(0.485, 0.456, 0.406) # ImageNet normalization constants, see
#' # https://pytorch.org/vision/stable/models.html
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
#' # ResNet-50 FPN
#' model <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE, score_thresh = 0.5,
#'                                         nms_thresh = 0.8, detections_per_img = 3)
#' model$eval()
#' torch::with_no_grad({pred <- model(batch_normalized)$detections[[1]]})
#'
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' keep <- seq_len(min(5, num_boxes))
#' boxes <- pred$boxes[keep, ]$view(c(-1, 4))
#' labels <- coco_classes(as.integer(pred$labels[keep]))
#' if (num_boxes > 0) {
#'   boxed <- draw_bounding_boxes(image, boxes, labels = labels)
#'   tensor_image_browse(boxed)
#' }
#'
#' # ResNet-50 FPN V2
#' model <- model_fasterrcnn_resnet50_fpn_v2(pretrained = TRUE)
#' model$eval()
#' torch::with_no_grad({pred <- model(batch_normalized)$detections[[1]]})
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' keep <- seq_len(min(5, num_boxes))
#' boxes <- pred$boxes[keep, ]$view(c(-1, 4))
#' labels <- coco_classes(as.integer(pred$labels[keep]))
#' if (num_boxes > 0) {
#'   boxed <- draw_bounding_boxes(image, boxes, labels = labels)
#'   tensor_image_browse(boxed)
#' }
#'
#' # MobileNet V3 Large FPN
#' batch <- image$unsqueeze(1)    # Add batch dimension (1, 3, H, W)
#'
#' model <- model_fasterrcnn_mobilenet_v3_large_fpn(
#'   pretrained = TRUE, score_thresh = 0.02, nms_thresh = 0.9, detections_per_img = 5
#' )
#' model$eval()
#' torch::with_no_grad({
#'   pred <- model(batch)$detections[[1]]
#' })
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' keep <- seq_len(min(5, num_boxes))
#' boxes <- pred$boxes[keep, ]$view(c(-1, 4))
#' labels <- coco_classes(as.integer(pred$labels[keep]))
#' if (num_boxes > 0) {
#'   boxed <- draw_bounding_boxes(image, boxes, labels = labels)
#'   tensor_image_browse(boxed)
#' }
#'
#' # MobileNet V3 Large 320 FPN
#' model <- model_fasterrcnn_mobilenet_v3_large_320_fpn(
#'   pretrained = TRUE, score_thresh = 0.02, nms_thresh = 0.8, detections_per_img = 5
#' )
#' model$eval()
#' torch::with_no_grad({pred <- model(batch)$detections[[1]]})
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' keep <- seq_len(min(5, num_boxes))
#' boxes <- pred$boxes[keep, ]$view(c(-1, 4))
#' labels <- coco_classes(as.integer(pred$labels[keep]))
#' if (num_boxes > 0) {
#'   boxed <- draw_bounding_boxes(image, boxes, labels = labels)
#'   tensor_image_browse(boxed)
#' }
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
                                          num_classes = 91,
                                          score_thresh = 0.05,
                                          nms_thresh = 0.5,
                                          detections_per_img = 100,
                                          ...) {
  backbone <- resnet_fpn_backbone(pretrained = pretrained)
  model <- fasterrcnn_model(backbone, num_classes = num_classes,
                            score_thresh = score_thresh,
                            nms_thresh = nms_thresh,
                            detections_per_img = detections_per_img)
  if (pretrained && num_classes != 91)
    cli_abort("Pretrained weights require num_classes = 91.")

  if (pretrained) {
    r <- rpn_model_urls$fasterrcnn_resnet50
    name <- "fasterrcnn_resnet50"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "fasterrcnn")
    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(.rename_fasterrcnn_state_dict(state_dict), strict = FALSE)
  }

  model
}


#' @describeIn model_fasterrcnn Faster R-CNN with ResNet-50 FPN V2
#' @export
model_fasterrcnn_resnet50_fpn_v2 <- function(pretrained = FALSE, progress = TRUE,
                                             num_classes = 91,
                                             score_thresh = 0.05,
                                             nms_thresh = 0.5,
                                             detections_per_img = 100,
                                             ...) {
  backbone <- resnet_fpn_backbone_v2(pretrained = pretrained)
  model <- fasterrcnn_model_v2(backbone, num_classes = num_classes,
                               score_thresh = score_thresh,
                               nms_thresh = nms_thresh,
                               detections_per_img = detections_per_img)

  if (pretrained && num_classes != 91)
    cli_abort("Pretrained weights require num_classes = 91.")

  if (pretrained) {
    r <- rpn_model_urls$fasterrcnn_resnet50_v2
    name <- "fasterrcnn_resnet50_v2"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "fasterrcnn")
    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }
    state_dict <- torch::load_state_dict(state_dict_path)
    state_dict <- .rename_fasterrcnn_v2_state_dict(state_dict)
    model_state <- model$state_dict()
    state_dict <- state_dict[names(state_dict) %in% names(model_state)]
    for (n in names(state_dict)) {
      if (!all(state_dict[[n]]$size() == model_state[[n]]$size())) {
        state_dict[[n]] <- model_state[[n]]
      }
    }
    missing <- setdiff(names(model_state), names(state_dict))
    if (length(missing) > 0) {
      for (n in missing) {
        state_dict[[n]] <- model_state[[n]]
      }
    }

    model$load_state_dict(state_dict)
  }

  model
}


#' @describeIn model_fasterrcnn Faster R-CNN with MobileNet V3 Large FPN
#' @export
model_fasterrcnn_mobilenet_v3_large_fpn <- function(pretrained = FALSE,
                                                    progress = TRUE,
                                                    num_classes = 91,
                                                    score_thresh = 0.05,
                                                    nms_thresh = 0.5,
                                                    detections_per_img = 100,
                                                    ...) {
  backbone <- mobilenet_v3_fpn_backbone(pretrained = pretrained)
  model <- fasterrcnn_mobilenet_model(backbone, num_classes = num_classes,
                                      score_thresh = score_thresh,
                                      nms_thresh = nms_thresh,
                                      detections_per_img = detections_per_img)

  if (pretrained && num_classes != 91)
    cli_abort("Pretrained weights require num_classes = 91.")

  if (pretrained) {
    r <- rpn_model_urls$fasterrcnn_mobilenet_v3_large
    name <- "fasterrcnn_mobilenet_v3_large"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "fasterrcnn")
    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(.rename_fasterrcnn_large_state_dict(state_dict), strict = FALSE)
  }

  model
}


#' @describeIn model_fasterrcnn Faster R-CNN with MobileNet V3 Large 320 FPN
#' @export
model_fasterrcnn_mobilenet_v3_large_320_fpn <- function(pretrained = FALSE,
                                                        progress = TRUE,
                                                        num_classes = 91,
                                                        score_thresh = 0.05,
                                                        nms_thresh = 0.5,
                                                        detections_per_img = 100,
                                                        ...) {
  backbone <- mobilenet_v3_320_fpn_backbone(pretrained = pretrained)
  model <- fasterrcnn_mobilenet_model(backbone, num_classes = num_classes,
                                      score_thresh = score_thresh,
                                      nms_thresh = nms_thresh,
                                      detections_per_img = detections_per_img)

  if (pretrained && num_classes != 91)
    cli_abort("Pretrained weights require num_classes = 91.")

  if (pretrained) {
    r <- rpn_model_urls$fasterrcnn_mobilenet_v3_large_320
    name <- "fasterrcnn_mobilenet_v3_large_320"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "fasterrcnn")
    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(.rename_fasterrcnn_large_state_dict(state_dict), strict = FALSE)
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

.rename_fasterrcnn_v2_state_dict <- function(state_dict) {
  . <- NULL # Nulling strategy for no visible binding check Note
  new_names <- names(state_dict) %>%
    # Fix FPN Sequential nesting: Conv layers (PyTorch 0.0 -> R 0)
    sub(pattern = "(inner_blocks\\.[0-3]\\.)0\\.0\\.", replacement = "\\10\\.", x = .) %>%
    sub(pattern = "(layer_blocks\\.[0-3]\\.)0\\.0\\.", replacement = "\\10\\.", x = .) %>%
    # BN layers (PyTorch 0.1 -> R 1)
    sub(pattern = "(inner_blocks\\.[0-3]\\.)0\\.1\\.", replacement = "\\11\\.", x = .) %>%
    sub(pattern = "(layer_blocks\\.[0-3]\\.)0\\.1\\.", replacement = "\\11\\.", x = .) %>%
    # Fix Box Head final linear layer (PyTorch 5 -> R 4)
    sub(pattern = "(roi_heads\\.box_head\\.)5\\.", replacement = "\\14\\.", x = .)

  # Recreate a list with renamed keys
  setNames(state_dict[names(state_dict)], new_names)
}
