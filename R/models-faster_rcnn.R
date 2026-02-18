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
      # The pretrained checkpoint stacks two Conv2d → BatchNorm2d → ReLU
      # blocks with bias disabled on the convolutions. Mirror that layout so
      # the parameter names and shapes line up with the weight file.
      block <- function() {
        nn_sequential(
          nn_conv2d(in_channels, in_channels, kernel_size = 3, padding = 1),
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
    "rpn_head_mobilenet",
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
generate_level_anchors <- function(h, w, stride, scales) {
  # Grid centers
  shift_x <- torch_arange(0.5, w - 0.5, 1.0) * stride
  shift_y <- torch_arange(0.5, h - 0.5, 1.0) * stride
  shifts <- torch_meshgrid(list(shift_x, shift_y), indexing = "xy")
  shift_grid <- torch_stack(list(shifts[[1]], shifts[[2]], shifts[[1]], shifts[[2]]), dim = 3)$unsqueeze(3)  # [H, W, 1, 4]

  # Anchor sizes (width/height)
  # Example: square anchors per scale
  anchor_sizes <- torch_tensor(scales) * stride  # [A]
  anchor_widths <- anchor_sizes
  anchor_heights <- anchor_sizes

  # Create base anchors [A, 4] (xc, yc, w, h)
  anchors <- torch_stack(list(
    torch_zeros_like(anchor_sizes),
    torch_zeros_like(anchor_sizes),
    anchor_widths,
    anchor_heights
  ), dim = 1)  # [A, 4]

  # Expand to [H, W, A, 4]
  anchors <- anchors$reshape(c(1, 1, -1, 4)) + shift_grid  # Broadcasting
  anchors
}

decode_boxes <- function(anchors, deltas) {
  widths <- anchors[, 3] - anchors[, 1]
  heights <- anchors[, 4] - anchors[, 2]
  ctr_x <- anchors[, 1] + widths / 2
  ctr_y <- anchors[, 2] + heights / 2

  dx <- deltas[, 1]
  dy <- deltas[, 2]
  dw <- deltas[, 3]
  dh <- deltas[, 4]

  pred_ctr_x <- ctr_x + dx * widths
  pred_ctr_y <- ctr_y + dy * heights
  pred_w <- torch::torch_exp(dw) * widths
  pred_h <- torch::torch_exp(dh) * heights

  x1 <- pred_ctr_x - pred_w / 2
  y1 <- pred_ctr_y - pred_h / 2
  x2 <- pred_ctr_x + pred_w / 2
  y2 <- pred_ctr_y + pred_h / 2

  torch::torch_stack(list(x1, y1, x2, y2), dim = 2)
}

generate_proposals <- function(features, rpn_out, image_size, strides,
                               score_thresh = 0.05, nms_thresh = 0.7) {
  device <- rpn_out$objectness[[1]]$device
  all_proposals <- torch::torch_empty(0L, 4L, device = device)
  all_scores <- torch::torch_empty(0L, device = device)

  for (i in seq_along(features)) {
    objectness <- rpn_out$objectness[[i]][1, , , ]
    deltas <- rpn_out$bbox_deltas[[i]][1, , , ]

    c(a, h, w) %<-% objectness$shape

    anchors <- generate_level_anchors(h, w, strides[[i]], scales = seq_len(a))
    anchors <- anchors$reshape(c(-1, 4))  # [H*W*A, 4]

    objectness <- objectness$sigmoid()$flatten() ## [H*W*A]
    deltas <- deltas$permute(c(2, 3, 1))$reshape(c(-1, 4))  # [H*W*A, 4]

    proposals <- decode_boxes(anchors, deltas)
    proposals <- clip_boxes_to_image(proposals, image_size)

    all_proposals <- torch::torch_cat(list(all_proposals, proposals), dim = 1L)
    all_scores <- torch::torch_cat(list(all_scores, objectness), dim = 1L)
  }

  scores <- all_scores$flatten()
  keep <- scores > score_thresh
  proposals <- all_proposals[keep, ]
  scores <- scores[keep]

  if (proposals$shape[1] > 0) {
    keep_idx <- nms(proposals, scores, nms_thresh)
    proposals <- proposals[keep_idx, ]
  } else {
    proposals <- torch::torch_empty(c(0, 4), device = device, dtype = torch_float32())
  }

  list(proposals = proposals)
}

roi_align_stub <- function(feature_map, proposals, output_size = c(7L, 7L)) {
  h <- as.integer(feature_map$shape[[3]])
  w <- as.integer(feature_map$shape[[4]])

  n <- proposals$size(1)
  pooled <- vector("list", n)

  for (i in seq_len(n)) {
    x1 <- max(1, min(as.numeric(proposals[i, 1]), w))
    y1 <- max(1, min(as.numeric(proposals[i, 2]), h))
    x2 <- max(x1 + 1, min(as.numeric(proposals[i, 3]), w))
    y2 <- max(y1 + 1, min(as.numeric(proposals[i, 4]), h))

    region <- feature_map[1, , y1:y2, x1:x2]
    pooled_feat <- torch::nnf_interpolate(
      region$unsqueeze(1),
      size = output_size,
      mode = "bilinear",
      align_corners = FALSE
    )[1, , , ]

    pooled[[i]] <- pooled_feat$reshape(-1)
  }

  torch::torch_stack(pooled)
}


roi_heads_module <- torch::nn_module(
  "Region‑of‑Interest Head",
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
  forward = function(features, proposals) {
    feature_maps <- features[c("p2", "p3", "p4", "p5")]
    pooled <- roi_align_stub(feature_maps[[1]], proposals)
    x <- self$box_head(pooled)
    self$box_predictor(x)
  }
  )


roi_heads_module_v2 <- torch::nn_module(
  "Region‑of‑Interest Head v2",
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
  forward = function(features, proposals) {
    pooled <- roi_align_stub(features[[1]], proposals)
    x <- self$box_head(pooled)
    self$box_predictor(x)
  }
)


fpn_module <- torch::nn_module(
  "Feature Pyramid Network",
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
    features <- self$backbone(images)
    rpn_out <- self$rpn(features)

    image_size <- as.integer(images$shape[3:4])
    props <- generate_proposals(features, rpn_out, image_size, c(4, 8, 16, 32),
                                score_thresh = self$score_thresh,
                                nms_thresh = self$nms_thresh)

    if (props$proposals$shape[1] == 0) {
      empty <- list(
        boxes = torch::torch_empty(c(0, 4)),
        labels = torch::torch_empty(c(0), dtype = torch::torch_long()),
        scores = torch::torch_empty(c(0))
      )
      return(list(features = features, detections = empty))
    }

    detections <- self$roi_heads(features, props$proposals)

    scores <- torch::nnf_softmax(detections$scores, dim = 2)
    max_scores <- torch::torch_max(scores, dim = 2)
    final_scores <- max_scores[[1]]
    final_labels <- max_scores[[2]]

    box_reg <- detections$boxes$view(c(-1, self$num_classes, 4))
    gather_idx <- final_labels$unsqueeze(2)$unsqueeze(3)$expand(c(-1, 1, 4))
    final_boxes <- box_reg$gather(2, gather_idx)$squeeze(2)

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
      final_boxes <- torch::torch_empty(c(0, 4))
      final_labels <- torch::torch_empty(c(0), dtype = torch::torch_long())
      final_scores <- torch::torch_empty(c(0))
    }

    list(
      features = features,
      detections = list(
        boxes = final_boxes,
        labels = final_labels,
        scores = final_scores
      )
    )
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
    features <- self$backbone(images)
    rpn_out <- self$rpn(features)

    image_size <- as.integer(images$shape[3:4])
    props <- generate_proposals(features, rpn_out, image_size, c(4, 8, 16, 32),
                                score_thresh = self$score_thresh,
                                nms_thresh = self$nms_thresh)

    if (props$proposals$shape[1] == 0) {
      empty <- list(
        boxes = torch::torch_empty(c(0, 4)),
        labels = torch::torch_empty(c(0), dtype = torch::torch_long()),
        scores = torch::torch_empty(c(0))
      )
      return(list(features = features, detections = empty))
    }

    detections <- self$roi_heads(features, props$proposals)

    scores <- torch::nnf_softmax(detections$scores, dim = 2)
    max_scores <- torch::torch_max(scores, dim = 2)
    final_scores <- max_scores[[1]]
    final_labels <- max_scores[[2]]

    box_reg <- detections$boxes$view(c(-1, self$num_classes, 4))
    gather_idx <- final_labels$unsqueeze(2)$unsqueeze(3)$expand(c(-1, 1, 4))
    final_boxes <- box_reg$gather(2, gather_idx)$squeeze(2)

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
      final_boxes <- torch::torch_empty(c(0, 4))
      final_labels <- torch::torch_empty(c(0), dtype = torch::torch_long())
      final_scores <- torch::torch_empty(c(0))
    }

    list(
      features = features,
      detections = list(
        boxes = final_boxes,
        labels = final_labels,
        scores = final_scores
      )
    )
  }
)


fpn_module_2level <- torch::nn_module(
  "feature pyramid network v2 level",
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
    features <- self$backbone(images)
    rpn_out <- self$rpn(features)

    image_size <- as.integer(images$shape[3:4])
    props <- generate_proposals(features, rpn_out, image_size, c(8, 16),
                                score_thresh = self$score_thresh,
                                nms_thresh = self$nms_thresh)

    if (props$proposals$shape[1] == 0) {
      empty <- list(
        boxes = torch::torch_empty(c(0, 4)),
        labels = torch::torch_empty(c(0), dtype = torch::torch_long()),
        scores = torch::torch_empty(c(0))
      )
      return(list(features = features, detections = empty))
    }

    detections <- self$roi_heads(features, props$proposals)

    scores <- nnf_softmax(detections$scores, dim = 2)
    max_scores <- torch_max(scores, dim = 2)
    final_scores <- max_scores[[1]]
    final_labels <- max_scores[[2]]

    box_reg <- detections$boxes$view(c(-1, self$num_classes, 4))
    gather_idx <- final_labels$unsqueeze(2)$unsqueeze(3)$expand(c(-1, 1, 4))
    final_boxes <- box_reg$gather(2, gather_idx)$squeeze(2)

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
      final_boxes <- torch::torch_empty(c(0, 4))
      final_labels <- torch::torch_empty(c(0), dtype = torch::torch_long())
      final_scores <- torch::torch_empty(c(0))
    }

    final <- list(boxes = final_boxes, labels = final_labels, scores = final_scores)
    list(features = features, detections = final)
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
#'              "e/ea/Morsan_Normande_vache.jpg/120px-Morsan_Normande_vache.jpg")
#' img <- base_loader(url)
#'
#' input <- img %>%
#'   transform_to_tensor() %>%
#'   transform_resize(c(520, 520))
#'
#' input <- img  %>%
#'   transform_normalize(norm_mean, norm_std)
#' batch <- input$unsqueeze(1)    # Add batch dimension (1, 3, H, W)
#'
#' # ResNet-50 FPN
#' model <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE, score_thresh = 0.5,
#'                                         nms_thresh = 0.8, detections_per_img = 3)
#' model$eval()
#' pred <- model(batch)$detections[[1]]
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' keep <- seq_len(min(5, num_boxes))
#' boxes <- pred$boxes[keep, ]$view(c(-1, 4))
#' labels <- coco_label(as.integer(pred$labels[keep]))
#' if (num_boxes > 0) {
#'   boxed <- draw_bounding_boxes(image, boxes, labels = labels)
#'   tensor_image_browse(boxed)
#' }
#'
#' # ResNet-50 FPN V2
#' model <- model_fasterrcnn_resnet50_fpn_v2(pretrained = TRUE)
#' model$eval()
#' pred <- model(batch)$detections[[1]]
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' keep <- seq_len(min(5, num_boxes))
#' boxes <- pred$boxes[keep, ]$view(c(-1, 4))
#' labels <- coco_label(as.integer(pred$labels[keep]))
#' if (num_boxes > 0) {
#'   boxed <- draw_bounding_boxes(img, boxes, labels = labels)
#'   tensor_image_browse(boxed)
#' }
#'
#' # MobileNet V3 Large FPN
#' model <- model_fasterrcnn_mobilenet_v3_large_fpn(pretrained = TRUE)
#' model$eval()
#' pred <- model(batch)$detections[[1]]
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' keep <- seq_len(min(5, num_boxes))
#' boxes <- pred$boxes[keep, ]$view(c(-1, 4))
#' labels <- coco_label(as.integer(pred$labels[keep]))
#' if (num_boxes > 0) {
#'   boxed <- draw_bounding_boxes(image, boxes, labels = labels)
#'   tensor_image_browse(boxed)
#' }
#'
#' # MobileNet V3 Large 320 FPN
#' model <- model_fasterrcnn_mobilenet_v3_large_320_fpn(pretrained = TRUE)
#' model$eval()
#' pred <- model(batch)$detections[[1]]
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' keep <- seq_len(min(5, num_boxes))
#' boxes <- pred$boxes[keep, ]$view(c(-1, 4))
#' labels <- coco_label(as.integer(pred$labels[keep]))]
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

    model_state <- model$state_dict()
    # TODO remove that model scalping
    # TODO will fail due to setdiff(names(model$modules), names(state_dict)), currently 221 discrepancies
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

    model$load_state_dict(state_dict, strict = TRUE)
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
