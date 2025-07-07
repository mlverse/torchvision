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
      self$ratios <- ratios
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

        anchors <- base_anchors$unsqueeze(2) + shifts$unsqueeze(1)
        anchors <- anchors$reshape(c(-1, 4))
        anchors_per_level[[i]] <- anchors
      }

      anchors_per_level
    }
  )
}

# Dummy proposal generator (replace with NMS and top-N selection)
generate_proposals <- function(anchors, objectness, bbox_deltas, image_size) {
  # Here we simply return anchors for now
  anchors
}

# ROI heads module with proper structure for pretrained weights
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
