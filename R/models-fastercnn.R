#' Faster R-CNN with ResNet-50 FPN
#'
#' Constructs a Faster R-CNN model with a ResNet-50 FPN backbone.
#'
#' @param pretrained Logical. If TRUE, loads pretrained weights from local file.
#' @param progress Logical. Show progress bar during download (unused).
#' @param num_classes Number of output classes (default: 91 for COCO).
#' @param ... Other arguments (unused).
#'
#' @return A `fasterrcnn_model` nn_module.
#' @export
model_fasterrcnn_resnet50_fpn <- function(pretrained = FALSE, progress = TRUE,
                                          num_classes = 91, ...) {
  backbone <- resnet_fpn_backbone(pretrained = pretrained)

  model <- fasterrcnn_model(backbone, num_classes = num_classes)

  if (pretrained) {
    local_path <- "tools/models/fasterrcnn_resnet50_fpn.pth"
    state_dict <- torch::load_state_dict(local_path)
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

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
      x <- self$conv1(x)
      x <- self$bn1(x)
      x <- self$relu(x)
      x <- self$maxpool(x)

      c2 <- self$layer1(x)
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
      )()
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

fpn_module <- function(in_channels, out_channels) {
  torch::nn_module(
    initialize = function() {
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
}

rpn_head <- function(in_channels, num_anchors = 3) {
  torch::nn_module(
    "rpn_head",
    initialize = function() {
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
}

anchor_generator_module <- function(scales = c(32, 64, 128, 256),
                                    ratios = c(0.5, 1.0, 2.0),
                                    strides = c(4, 8, 16, 32)) {
  torch::nn_module(
    initialize = function() {
      self$scales <- scales
      self$ratios  <- ratios
      self$strides <- strides

      self$num_anchors <- length(ratios)
      self$base_anchors <- lapply(seq_along(scales), function(i) {
        base_size <- scales[i]
        anchors <- list()

        for (ratio in ratios) {
          for (scale in 1) {  # Single scale per level
            w <- base_size * sqrt(1 / ratio)
            h <- base_size * sqrt(ratio)
            anchors[[length(anchors) + 1]] <- c(-w / 2, -h / 2, w / 2, h / 2)
          }
        }

        torch::torch_tensor(
          do.call(rbind, anchors),
          dtype = torch::torch_float()
        )
      })
    },

    forward = function(features, image_size) {
      anchors_per_level <- list()

      for (i in seq_along(features)) {
        feature_map <- features[[i]]
        stride <- self$strides[i]
        base_anchors <- self$base_anchors[[i]]

        fm_height <- feature_map$shape[3]
        fm_width <- feature_map$shape[4]

        shifts_x <- torch::torch_arange(0, fm_width - 1, dtype = torch::torch_float()) * stride
        shifts_y <- torch::torch_arange(0, fm_height - 1, dtype = torch::torch_float()) * stride

        shift_grid <- torch::torch_meshgrid(list(shifts_y, shifts_x), indexing = "ij")
        shift_y <- shift_grid[[1]]$reshape(c(-1))
        shift_x <- shift_grid[[2]]$reshape(c(-1))

        shifts <- torch::torch_stack(list(shift_x, shift_y, shift_x, shift_y), dim = 2)

        anchors <- base_anchors$unsqueeze(1) + shifts$unsqueeze(2)
        anchors <- anchors$reshape(c(-1, 4))
        anchors_per_level[[i]] <- anchors
      }

      anchors_per_level
    }
  )
}

# Helper functions for proposal generation
apply_deltas_to_anchors <- function(anchors, deltas) {
  # Simple implementation - in practice you'd want proper box regression
  # This is a placeholder that just returns the anchors
  anchors
}

remove_small_boxes <- function(boxes, min_size = 1) {
  # Simple implementation - keep all boxes for now
  torch::torch_arange(1, boxes$shape[1], dtype = torch::torch_long())
}

clip_boxes_to_image <- function(boxes, size) {
  # Simple clipping implementation
  boxes$clamp(min = 0, max = min(size))
}

nms <- function(boxes, scores, iou_threshold = 0.7) {
  # Simple NMS implementation - in practice use torchvision::nms
  # For now, just return top indices
  num_keep <- min(100, scores$shape[1])
  topk_result <- scores$topk(k = num_keep)
  topk_result[[2]]  # Return indices
}

roi_align <- function(feature_maps, proposals, output_size) {
  # Simplified ROI align - in practice you'd use proper implementation
  # Return dummy pooled features for now
  num_proposals <- proposals$shape[1]
  num_channels <- feature_maps$p2$shape[2]
  torch::torch_randn(c(num_proposals, num_channels, output_size[1], output_size[2]))
}

generate_proposals <- function(anchors, objectness, bbox_deltas, image_size) {
  # Filter out any empty tensors
  anchors <- Filter(function(x) x$numel() > 0 && x$ndim == 2 && x$shape[2] == 4, anchors)
  objectness <- Filter(function(x) x$numel() > 0, objectness)
  bbox_deltas <- Filter(function(x) x$numel() > 0, bbox_deltas)

  # Reshape anchors if needed
  anchors <- lapply(anchors, function(x) {
    if (x$ndim == 1) x$reshape(c(-1, 4)) else x
  })

  # Reshape objectness: (1, A, H, W) → (N, 1)
  objectness <- lapply(objectness, function(x) {
    if (x$ndim == 4) {
      x <- x$permute(c(1, 3, 4, 2))  # (1, H, W, A)
      x <- x$reshape(c(-1, 1))
    }
    x
  })

  # Reshape bbox_deltas: (1, A*4, H, W) → (N, 4)
  bbox_deltas <- lapply(bbox_deltas, function(x) {
    if (x$ndim == 4) {
      A4 <- x$shape[2]
      A <- A4 %/% 4
      x <- x$reshape(c(1, A, 4, x$shape[3], x$shape[4]))  # (1, A, 4, H, W)
      x <- x$permute(c(1, 4, 5, 2, 3))                   # (1, H, W, A, 4)
      x <- x$reshape(c(-1, 4))                           # (N, 4)
    }
    x
  })

  # Double check everything is still valid
  if (length(anchors) == 0 || length(objectness) == 0 || length(bbox_deltas) == 0) {
    stop("Empty input tensors after preprocessing.")
  }

  # Concatenate - FIX: Use dim = 1 for R's 1-based indexing
  flat_anchors <- torch::torch_cat(anchors, dim = 1)
  flat_objectness <- torch::torch_cat(objectness, dim = 1)$squeeze()
  flat_deltas <- torch::torch_cat(bbox_deltas, dim = 1)

  # Apply deltas
  boxes <- apply_deltas_to_anchors(flat_anchors, flat_deltas)

  # Remove small boxes
  keep <- remove_small_boxes(boxes, min_size = 1)
  boxes <- boxes[keep, ]
  scores <- flat_objectness[keep]

  # Clip to image
  boxes <- clip_boxes_to_image(boxes, size = image_size)

  # Pre-NMS top-k
  num_topk <- min(6000, scores$shape[1])
  if (num_topk > 0) {
    topk <- scores$topk(k = num_topk)
    indices <- topk[[2]]
    boxes <- boxes[indices, ]
    scores <- scores[indices]
  }

  # NMS
  keep <- nms(boxes, scores, iou_threshold = 0.7)
  boxes <- boxes[keep, ]
  scores <- scores[keep]

  # Final topk after NMS
  num_keep <- min(1000, scores$shape[1])
  if (num_keep > 0) {
    topk <- scores$topk(k = num_keep)
    indices <- topk[[2]]
    boxes <- boxes[indices, ]
  }

  # Add batch idx
  N <- boxes$shape[1]
  batch_indices <- torch::torch_zeros(N, dtype = torch::torch_long())
  rois <- torch::torch_cat(list(batch_indices$unsqueeze(2), boxes), dim = 2)

  return(rois)
}


# ROI heads module with proper structure for pretrained weights
roi_heads_module <- function(num_classes = 91) {
  torch::nn_module(
    initialize = function() {
      self$box_head <- torch::nn_module(
        initialize = function() {
          self$fc6 <- torch::nn_linear(256 * 7 * 7, 1024)
          self$fc7 <- torch::nn_linear(1024, 1024)
        },
        forward = function(x) {
          x <- torch::nnf_relu(self$fc6(x))
          x <- torch::nnf_relu(self$fc7(x))
          x
        }
      )()

      self$box_predictor <- torch::nn_module(
        initialize = function() {
          self$cls_score <- torch::nn_linear(1024, num_classes)
          self$bbox_pred <- torch::nn_linear(1024, num_classes * 4)
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
      # Use feature maps from backbone
      feature_maps <- features[c("p2", "p3", "p4", "p5")]

      # Apply RoI Align
      pooled <- roi_align(feature_maps, proposals, output_size = c(7, 7))

      # Flatten for FC layers
      pooled <- pooled$reshape(c(pooled$shape[1], -1))  # (N, C*7*7)

      x <- self$box_head(pooled)
      predictions <- self$box_predictor(x)

      predictions
    }
  )
}

fasterrcnn_model <- function(backbone, num_classes) {
  torch::nn_module(
    initialize = function() {
      self$backbone <- backbone

      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head(in_channels = backbone$out_channels)()
        },
        forward = function(features) {
          self$head(features)
        }
      )()

      self$anchor_generator <- anchor_generator_module()()

      # Use the roi_heads_module instead of inline definition
      self$roi_heads <- roi_heads_module(num_classes = num_classes)()
    },

    forward = function(images) {
      features <- self$backbone(images)
      rpn_out <- self$rpn(features)

      anchors <- self$anchor_generator(features, image_size = c(800, 800))

      proposals <- generate_proposals(
        anchors,
        rpn_out$objectness,
        rpn_out$bbox_deltas,
        image_size = c(800, 800)
      )

      detections <- self$roi_heads(features, proposals)

      # Postprocessing: get max scores and class labels
      scores <- torch::nnf_softmax(detections$scores, dim = 2)
      max_scores <- torch::torch_max(scores, dim = 2)
      final_scores <- max_scores$values
      final_labels <- max_scores$indices

      # For simplicity, assume detections$boxes already contains the correct boxes
      # In a full implementation, you'd need proper box regression decoding
      final_boxes <- detections$boxes

      # If boxes has more than 4 columns, take only the first 4
      if (final_boxes$shape[2] > 4) {
        final_boxes <- final_boxes[, 1:4]
      }

      # Fix: Ensure we're working with torch tensors throughout
      keep <- final_scores > 0.5

      # Use torch methods for sum and item operations
      num_detections <- torch::torch_sum(keep)$item()

      # Handle case where no detections meet the threshold
      if (num_detections > 0) {
        final_boxes <- final_boxes[keep, ]
        final_labels <- final_labels[keep]
        final_scores <- final_scores[keep]
      } else {
        # Return empty tensors if no detections
        final_boxes <- torch::torch_empty(c(0, 4))
        final_labels <- torch::torch_empty(c(0), dtype = torch::torch_long())
        final_scores <- torch::torch_empty(c(0))
      }

      list(
        features = features,
        proposals = proposals,
        detections = list(
          boxes = final_boxes,
          labels = final_labels,
          scores = final_scores
        )
      )
    }
  )()
}
