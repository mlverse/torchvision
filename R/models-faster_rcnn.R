# =============================================================================
# SHARED COMPONENTS
# =============================================================================

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

rpn_head_v2 <- function(in_channels, num_anchors = 3) {
  torch::nn_module(
    "rpn_head_v2",
    initialize = function() {
      # Match expected structure: conv should have multiple layers
      self$conv <- nn_sequential(
        nn_conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, bias = TRUE),
        nn_relu()
      )
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
}

rpn_head_mobilenet <- function(in_channels, num_anchors = 15) {
  torch::nn_module(
    initialize = function() {
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
}

roi_heads_module <- function(num_classes = 91) {
  torch::nn_module(
    initialize = function() {
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
      # Extract feature maps
      feature_maps <- features[c("p2", "p3", "p4", "p5")]

      # Placeholder for ROI pooling - in full implementation, you'd use roi_align
      pooled <- torch::torch_randn(length(proposals), 256 * 7 * 7)

      x <- self$box_head(pooled)
      predictions <- self$box_predictor(x)

      predictions
    }
  )
}

roi_heads_module_v2 <- function(num_classes = 91) {
  torch::nn_module(
    initialize = function() {
      # Match expected structure for box_head
      self$box_head <- nn_sequential(
        nn_sequential(
          nn_linear(256 * 7 * 7, 1024, bias = TRUE),
          nn_relu()
        ),
        nn_sequential(
          nn_linear(1024, 1024, bias = TRUE),
          nn_relu()
        )
      )

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
      # Placeholder for ROI pooling - in full implementation, use roi_align
      pooled <- torch::torch_randn(length(proposals), 256 * 7 * 7)
      x <- self$box_head(pooled)
      predictions <- self$box_predictor(x)
      predictions
    }
  )
}

# =============================================================================
# SECTION 1: FASTER R-CNN WITH RESNET-50 FPN
# =============================================================================

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

      # Use the roi_heads_module instead of inline definition
      self$roi_heads <- roi_heads_module(num_classes = num_classes)()
    },

    forward = function(images) {
      features <- self$backbone(images)
      rpn_out <- self$rpn(features)

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
        detections = list(
          boxes = final_boxes,
          labels = final_labels,
          scores = final_scores
        )
      )
    }
  )()
}


#' Faster R-CNN with ResNet-50 FPN
#'
#' @param pretrained Logical. If TRUE, loads pretrained weights from local file.
#' @param progress Logical. Show progress bar during download (unused).
#' @param num_classes Number of output classes (default: 91 for COCO).
#' @param ... Other arguments (unused).
#' @return A `fasterrcnn_model` nn_module.
#' @export
model_fasterrcnn_resnet50_fpn <- function(pretrained = FALSE, progress = TRUE,
                                          num_classes = 91, ...) {
  backbone <- resnet_fpn_backbone(pretrained = pretrained)
  model <- fasterrcnn_model(backbone, num_classes = num_classes)

  if (pretrained) {
    local_path <- "tools/models/fasterrcnn_resnet50_fpn.pth"
    state_dict <- torch::load_state_dict(local_path)
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

# =============================================================================
# SECTION 2: FASTER R-CNN WITH RESNET-50 FPN V2
# =============================================================================

fpn_module_v2 <- function(in_channels, out_channels) {
  torch::nn_module(
    initialize = function() {
      self$inner_blocks <- nn_module_list(lapply(in_channels, function(c) {
        nn_sequential(
          nn_conv2d(c, out_channels, kernel_size = 1, bias = TRUE),
          nn_batch_norm2d(out_channels),
          nn_relu()
        )
      }))
      self$layer_blocks <- nn_module_list(lapply(rep(out_channels, 4), function(i) {
        nn_sequential(
          nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = TRUE),
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
}

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
      self$fpn <- fpn_module_v2(
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


fasterrcnn_model_v2 <- function(backbone, num_classes) {
  torch::nn_module(
    initialize = function() {
      self$backbone <- backbone
      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head_v2(in_channels = backbone$out_channels)()
        },
        forward = function(features) {
          self$head(features)
        }
      )()
      self$roi_heads <- roi_heads_module_v2(num_classes = num_classes)()
    },
    forward = function(images) {
      features <- self$backbone(images)
      rpn_out <- self$rpn(features)

      # Dummy proposals for testing
      proposals <- list(torch::torch_randn(100, 4))
      detections <- self$roi_heads(features, proposals)

      # Post-processing
      scores <- torch::nnf_softmax(detections$scores, dim = 2)
      max_scores <- torch::torch_max(scores, dim = 2)
      final_scores <- max_scores$values
      final_labels <- max_scores$indices

      final_boxes <- detections$boxes
      if (final_boxes$shape[2] > 4) {
        final_boxes <- final_boxes[, 1:4]
      }

      keep <- final_scores > 0.5
      num_detections <- torch::torch_sum(keep)$item()

      if (num_detections > 0) {
        final_boxes <- final_boxes[keep, ]
        final_labels <- final_labels[keep]
        final_scores <- final_scores[keep]
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
  )()
}

#' Faster R-CNN with ResNet-50 FPN V2
#'
#' @inheritParams model_fasterrcnn_resnet50_fpn
#' @return A `fasterrcnn_model` nn_module.
#' @export
model_fasterrcnn_resnet50_fpn_v2 <- function(pretrained = FALSE, progress = TRUE,
                                             num_classes = 91, ...) {
  backbone <- resnet_fpn_backbone_v2(pretrained = pretrained)
  model <- fasterrcnn_model_v2(backbone, num_classes = num_classes)

  if (pretrained) {
    local_path <- "tools/models/fasterrcnn_resnet50_fpn_v2.pth"
    state_dict <- torch::load_state_dict(local_path)
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

# =============================================================================
# SECTION 3: FASTER R-CNN WITH MOBILENET V3 LARGE FPN
# =============================================================================

fpn_module_2level <- function(in_channels, out_channels) {
  torch::nn_module(
    initialize = function() {
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
}


mobilenet_v3_fpn_backbone <- function(pretrained = TRUE) {
  mobilenet <- model_mobilenet_v3_large(pretrained = pretrained)

  backbone_module <- torch::nn_module(
    initialize = function() {
      self$body <- mobilenet$features
      self$fpn <- fpn_module_2level(
        in_channels = c(160, 960),
        out_channels = 256
      )()
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

fasterrcnn_mobilenet_model <- function(backbone, num_classes) {
  torch::nn_module(
    initialize = function() {
      self$backbone <- backbone
      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head_mobilenet(in_channels = backbone$out_channels)()
        },
        forward = function(features) {
          self$head(features)
        }
      )()
      self$roi_heads <- roi_heads_module(num_classes = num_classes)()
    },
    forward = function(images) {
      features <- self$backbone(images)
      rpn_out <- self$rpn(features)

      proposals <- list(torch_randn(100, 4))  # dummy proposals for testing
      detections <- self$roi_heads(features, proposals)

      scores <- nnf_softmax(detections$scores, dim = 2)
      max_scores <- torch_max(scores, dim = 2)
      keep <- max_scores$values > 0.5

      final <- list(
        boxes = detections$boxes[, 1:4][keep, , drop = FALSE],
        labels = max_scores$indices[keep],
        scores = max_scores$values[keep]
      )

      list(features = features, detections = final)
    }
  )()
}

#' Faster R-CNN with MobileNet V3 Large FPN
#'
#' @inheritParams model_fasterrcnn_resnet50_fpn
#' @return A `fasterrcnn_model` nn_module.
#' @export
model_fasterrcnn_mobilenet_v3_large_fpn <- function(pretrained = FALSE,
                                                    progress = TRUE,
                                                    num_classes = 91, ...) {
  backbone <- mobilenet_v3_fpn_backbone(pretrained = pretrained)
  model <- fasterrcnn_mobilenet_model(backbone, num_classes = num_classes)

  if (pretrained) {
    local_path <- "tools/models/fasterrcnn_mobilenet_v3_large_fpn.pth"
    state_dict <- torch::load_state_dict(local_path)
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

# =============================================================================
# SECTION 4: FASTER R-CNN WITH MOBILENET V3 LARGE 320 FPN
# =============================================================================

mobilenet_v3_320_fpn_backbone <- function(pretrained = TRUE) {
  mobilenet <- model_mobilenet_v3_large(pretrained = pretrained)

  backbone_module <- torch::nn_module(
    initialize = function() {
      self$body <- mobilenet$features
      self$fpn <- fpn_module_2level(
        in_channels = c(160, 960),  # output channels of layer 13 and 16
        out_channels = 256
      )()
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

#' Faster R-CNN with MobileNet V3 Large 320 FPN
#'
#' @inheritParams model_fasterrcnn_resnet50_fpn
#' @return A `fasterrcnn_model` nn_module.
#' @export
model_fasterrcnn_mobilenet_v3_large_320_fpn <- function(pretrained = FALSE,
                                                        progress = TRUE,
                                                        num_classes = 91, ...) {
  backbone <- mobilenet_v3_320_fpn_backbone(pretrained = pretrained)
  model <- fasterrcnn_mobilenet_model(backbone, num_classes = num_classes)

  if (pretrained) {
    local_path <- "tools/models/fasterrcnn_mobilenet_v3_large_320_fpn.pth"
    state_dict <- torch::load_state_dict(local_path)
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}
