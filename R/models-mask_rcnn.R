#' @include models-faster_rcnn.R
#' @importFrom torch nn_conv_transpose2d torch_int32
NULL

# Mask R-CNN Implementation
# Instance Segmentation model extending Faster R-CNN with mask prediction

# Mask Head Module - Predicts segmentation masks for detected objects
mask_head_module <- function(num_classes = 91) {
  torch::nn_module(
    "mask_head",
    initialize = function() {
      # 4 convolutional layers for feature extraction
      self$conv1 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)
      self$conv2 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)
      self$conv3 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)
      self$conv4 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)

      # Deconvolution layer to upsample from 14x14 to 28x28
      self$deconv <- nn_conv_transpose2d(256, 256, kernel_size = 2, stride = 2)

      # Final 1x1 conv for class-specific mask logits
      self$mask_fcn_logits <- nn_conv2d(256, num_classes, kernel_size = 1)
    },
    forward = function(x) {
      x <- nnf_relu(self$conv1(x))
      x <- nnf_relu(self$conv2(x))
      x <- nnf_relu(self$conv3(x))
      x <- nnf_relu(self$conv4(x))
      x <- nnf_relu(self$deconv(x))
      self$mask_fcn_logits(x)
    }
  )
}

# Mask Head Module V2 - With batch normalization
mask_head_module_v2 <- function(num_classes = 91) {
  torch::nn_module(
    "mask_head_v2",
    initialize = function() {
      # Convolutional blocks with batch normalization
      conv_block <- function() {
        nn_sequential(
          nn_conv2d(256, 256, kernel_size = 3, padding = 1, bias = FALSE),
          nn_batch_norm2d(256),
          nn_relu()
        )
      }

      self$conv1 <- conv_block()
      self$conv2 <- conv_block()
      self$conv3 <- conv_block()
      self$conv4 <- conv_block()

      # Deconvolution with batch norm
      self$deconv <- nn_sequential(
        nn_conv_transpose2d(256, 256, kernel_size = 2, stride = 2, bias = FALSE),
        nn_batch_norm2d(256),
        nn_relu()
      )

      # Final 1x1 conv for class-specific mask logits
      self$mask_fcn_logits <- nn_conv2d(256, num_classes, kernel_size = 1)
    },
    forward = function(x) {
      x <- self$conv1(x)
      x <- self$conv2(x)
      x <- self$conv3(x)
      x <- self$conv4(x)
      x <- self$deconv(x)
      self$mask_fcn_logits(x)
    }
  )
}

#' ROI Align for Mask Prediction
#'
#' Extracts fixed-size feature maps from regions of interest for mask prediction.
#' Returns a 4D tensor suitable for the mask head (unlike roi_align_stub which flattens).
#'
#' @param feature_map Tensor of shape (1, C, H, W) - Feature map from backbone
#' @param proposals Tensor of shape (N, 4) - Region proposals as (x1, y1, x2, y2)
#' @param output_size Integer vector of length 2 - Output spatial dimensions (default: c(14L, 14L))
#'
#' @return Tensor of shape (N, C, output_size\[1\], output_size\[2\])
#'
#' @noRd
roi_align_masks <- function(feature_map, proposals, output_size = c(14L, 14L)) {

  # 1. Retrieve dimensions
  n <- proposals$size(1)
  c_channels <- feature_map$size(2)
  h <- feature_map$size(3)
  w <- feature_map$size(4)

  if (n == 0) {
    return(torch_zeros(c(0, c_channels, output_size[1], output_size[2]),
                       dtype = feature_map$dtype,
                       device = feature_map$device))
  }

  # 2. Vectorized Coordinate Handling
  # Perform clamping on the GPU first to avoid multiple transfers
  bounds <- torch_tensor(c(1, 1, 1, 1, w, h, w, h),
                         dtype = proposals$dtype,
                         device = proposals$device)

  proposals_clamped <- torch_clamp(proposals,
                                   min = bounds[1:4]$view(c(1, 4)),
                                   max = bounds[5:8]$view(c(1, 4)))

  # Transfer coordinates to CPU in one go for R indexing
  boxes <- as_array(proposals_clamped$to(dtype = torch_int32(), device = "cpu"))

  # Ensure minimum boxes size of 1 pixel (questionable)
  # boxes[, 3] <- pmax(boxes[, 1] + 1L, boxes[, 3])
  # boxes[, 4] <- pmax(boxes[, 2] + 1L, boxes[, 4])

  # 3. Optimized loop
  # Store the full 4D output (1, C, H, W) to avoid slicing inside the loop.
  pooled <- vector("list", n)

  # We process each proposal individually (to avoid padding overhead)
  # and write directly into the pre-allocated result tensor.
  for (i in seq_len(n)) {
    # Get 1-based integer coordinates
    region <- feature_map[1, , boxes[i, 2]:boxes[i, 4], boxes[i, 1]:boxes[i, 3]]

    # nnf_interpolate returns (1, C, 14, 14)
    pooled[[i]] <- nnf_interpolate(
      region$unsqueeze(1),
      size = output_size,
      mode = "bilinear",
      align_corners = FALSE
    )
  }

  # Stack list of (1, C, H, W) tensors -> (N, 1, C, H, W)
  stacked <- torch_stack(pooled)

  # Remove the extra dimension (dim=2) to get (N, C, H, W)
  stacked$squeeze(2)
}


# Mask R-CNN Model - Extends Faster R-CNN with mask prediction
maskrcnn_model <- function(backbone, num_classes,
                           score_thresh = 0.05,
                           nms_thresh = 0.5,
                           detections_per_img = 100) {
  torch::nn_module(
    initialize = function() {
      self$backbone <- backbone

      # Store configurable detection parameters
      self$score_thresh <- score_thresh
      self$nms_thresh <- nms_thresh
      self$detections_per_img <- detections_per_img

      # RPN (Region Proposal Network)
      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head(in_channels = backbone$out_channels)()
        },
        forward = function(features) {
          self$head(features)
        }
      )()

      # ROI heads for box prediction
      self$roi_heads <- roi_heads_module(num_classes = num_classes)()

      # Mask head for mask prediction
      self$mask_head <- mask_head_module(num_classes = num_classes)()
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
          scores = torch::torch_empty(c(0)),
          masks = torch::torch_empty(c(0, 28, 28))
        )
        return(list(features = features, detections = empty))
      }

      # Limit proposals to avoid slow ROI pooling (common practice in detection)
      max_proposals <- 1000
      if (props$proposals$shape[1] > max_proposals) {
        props$proposals <- props$proposals[1:max_proposals, ]
      }

      # Box predictions
      detections <- self$roi_heads(features, props$proposals)

      scores <- torch::nnf_softmax(detections$scores, dim = 2)
      max_scores <- torch::torch_max(scores, dim = 2)
      final_scores <- max_scores[[1]]
      final_labels <- max_scores[[2]]

      box_reg <- detections$boxes$view(c(-1, num_classes, 4))
      gather_idx <- final_labels$unsqueeze(2)$unsqueeze(3)$expand(c(-1, 1, 4))
      final_boxes <- box_reg$gather(2, gather_idx)$squeeze(2)

      # Filter by score threshold
      keep <- final_scores > self$score_thresh
      num_detections <- torch::torch_sum(keep)$item()

      if (num_detections > 0) {
        final_boxes <- final_boxes[keep, ]
        final_labels <- final_labels[keep]
        final_scores <- final_scores[keep]
        kept_proposals <- props$proposals[keep, ]

        # Apply NMS to remove overlapping detections
        if (final_boxes$shape[1] > 1) {
          nms_keep <- nms(final_boxes, final_scores, self$nms_thresh)
          final_boxes <- final_boxes[nms_keep, ]
          final_labels <- final_labels[nms_keep]
          final_scores <- final_scores[nms_keep]
          kept_proposals <- kept_proposals[nms_keep, ]
        }

        # Limit detections per image
        n_det <- final_scores$shape[1]
        if (n_det > self$detections_per_img) {
          top_k <- torch::torch_topk(final_scores, self$detections_per_img)
          top_idx <- top_k[[2]]
          final_boxes <- final_boxes[top_idx, ]
          final_labels <- final_labels[top_idx]
          final_scores <- final_scores[top_idx]
          kept_proposals <- kept_proposals[top_idx, ]
        }

        # Predict masks for kept detections
        mask_features <- roi_align_masks(features[[1]], kept_proposals, output_size = c(14L, 14L))
        mask_logits <- self$mask_head(mask_features)  # Shape: (N, num_classes, 28, 28)

        # Extract masks for predicted classes
        n_kept <- final_labels$shape[1]
        final_masks <- torch::torch_zeros(c(n_kept, 28, 28))

        for (i in seq_len(n_kept)) {
          class_idx <- as.integer(final_labels[i]$item())
          final_masks[i, , ] <- mask_logits[i, class_idx, , ]
        }

        # Apply sigmoid to get probabilities
        final_masks <- torch::torch_sigmoid(final_masks)

      } else {
        final_boxes <- torch::torch_empty(c(0, 4))
        final_labels <- torch::torch_empty(c(0), dtype = torch::torch_long())
        final_scores <- torch::torch_empty(c(0))
        final_masks <- torch::torch_empty(c(0, 28, 28))
      }

      list(
        features = features,
        detections = list(
          boxes = final_boxes,
          labels = final_labels,
          scores = final_scores,
          masks = final_masks
        )
      )
    }
  )
}

# Mask R-CNN Model V2 - With batch normalization
maskrcnn_model_v2 <- function(backbone, num_classes,
                              score_thresh = 0.05,
                              nms_thresh = 0.5,
                              detections_per_img = 100) {
  torch::nn_module(
    initialize = function() {
      self$backbone <- backbone

      # Store configurable detection parameters
      self$score_thresh <- score_thresh
      self$nms_thresh <- nms_thresh
      self$detections_per_img <- detections_per_img

      # RPN with V2 head
      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head_v2(in_channels = backbone$out_channels)()
        },
        forward = function(features) {
          self$head(features)
        }
      )()

      # ROI heads V2 for box prediction
      self$roi_heads <- roi_heads_module_v2(num_classes = num_classes)()

      # Mask head V2 for mask prediction
      self$mask_head <- mask_head_module_v2(num_classes = num_classes)()
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
          scores = torch::torch_empty(c(0)),
          masks = torch::torch_empty(c(0, 28, 28))
        )
        return(list(features = features, detections = empty))
      }

      # Limit proposals to avoid slow ROI pooling (common practice in detection)
      max_proposals <- 1000
      if (props$proposals$shape[1] > max_proposals) {
        props$proposals <- props$proposals[1:max_proposals, ]
      }

      # Box predictions
      detections <- self$roi_heads(features, props$proposals)

      scores <- torch::nnf_softmax(detections$scores, dim = 2)
      max_scores <- torch::torch_max(scores, dim = 2)
      final_scores <- max_scores[[1]]
      final_labels <- max_scores[[2]]

      box_reg <- detections$boxes$view(c(-1, num_classes, 4))
      gather_idx <- final_labels$unsqueeze(2)$unsqueeze(3)$expand(c(-1, 1, 4))
      final_boxes <- box_reg$gather(2, gather_idx)$squeeze(2)

      # Filter by score threshold
      keep <- final_scores > self$score_thresh
      num_detections <- torch::torch_sum(keep)$item()

      if (num_detections > 0) {
        final_boxes <- final_boxes[keep, ]
        final_labels <- final_labels[keep]
        final_scores <- final_scores[keep]
        kept_proposals <- props$proposals[keep, ]

        # Apply NMS to remove overlapping detections
        if (final_boxes$shape[1] > 1) {
          nms_keep <- nms(final_boxes, final_scores, self$nms_thresh)
          final_boxes <- final_boxes[nms_keep, ]
          final_labels <- final_labels[nms_keep]
          final_scores <- final_scores[nms_keep]
          kept_proposals <- kept_proposals[nms_keep, ]
        }

        # Limit detections per image
        n_det <- final_scores$shape[1]
        if (n_det > self$detections_per_img) {
          top_k <- torch::torch_topk(final_scores, self$detections_per_img)
          top_idx <- top_k[[2]]
          final_boxes <- final_boxes[top_idx, ]
          final_labels <- final_labels[top_idx]
          final_scores <- final_scores[top_idx]
          kept_proposals <- kept_proposals[top_idx, ]
        }

        # Predict masks for kept detections
        mask_features <- roi_align_masks(features[[1]], kept_proposals, output_size = c(14L, 14L))
        mask_logits <- self$mask_head(mask_features)  # Shape: (N, num_classes, 28, 28)

        # Extract masks for predicted classes
        n_kept <- final_labels$shape[1]
        final_masks <- torch::torch_zeros(c(n_kept, 28, 28))

        for (i in seq_len(n_kept)) {
          class_idx <- as.integer(final_labels[i]$item())
          final_masks[i, , ] <- mask_logits[i, class_idx, , ]
        }

        # Apply sigmoid to get probabilities
        final_masks <- torch::torch_sigmoid(final_masks)

      } else {
        final_boxes <- torch::torch_empty(c(0, 4))
        final_labels <- torch::torch_empty(c(0), dtype = torch::torch_long())
        final_scores <- torch::torch_empty(c(0))
        final_masks <- torch::torch_empty(c(0, 28, 28))
      }

      list(
        features = features,
        detections = list(
          boxes = final_boxes,
          labels = final_labels,
          scores = final_scores,
          masks = final_masks
        )
      )
    }
  )
}

#' Mask R-CNN Models
#'
#' Construct Mask R-CNN model variants for instance segmentation task.
#' Mask R-CNN extends Faster R-CNN by adding a mask prediction branch that
#' outputs segmentation masks for each detected object.
#'
#' @param pretrained Logical. If TRUE, loads pretrained weights from local file.
#' @param progress Logical. Show progress bar during download (unused).
#' @param num_classes Number of output classes (default: 91 for COCO).
#' @param score_thresh Numeric. Minimum score threshold for detections (default: 0.05).
#' @param nms_thresh Numeric. Non-Maximum Suppression (NMS) IoU threshold for removing overlapping boxes (default: 0.5).
#' @param detections_per_img Integer. Maximum number of detections per image (default: 100).
#' @param ... Other arguments (unused).
#' @return A `maskrcnn_model` nn_module.
#'
#' @section Task:
#' Instance segmentation over images with bounding boxes, class labels, and segmentation masks.
#'
#' @section Input Format:
#' Input images should be `torch_tensor`s of shape
#' \verb{(batch_size, 3, H, W)} where `H` and `W` are typically around 800.
#'
#' @section Output Format:
#' Returns a list with:
#' \itemize{
#'   \item `features`: Feature maps from the backbone
#'   \item `detections`: List containing:
#'     \itemize{
#'       \item `boxes`: Bounding boxes (N, 4)
#'       \item `labels`: Class labels (N)
#'       \item `scores`: Confidence scores (N)
#'       \item `masks`: Segmentation masks (N, 28, 28)
#'     }
#' }
#'
#' @section Available Models:
#' \itemize{
#' \item `model_maskrcnn_resnet50_fpn()`
#' \item `model_maskrcnn_resnet50_fpn_v2()`
#' }
#'
#' @examples
#' \dontrun{
#' library(magrittr)
#' norm_mean <- c(0.485, 0.456, 0.406)
#' norm_std  <- c(0.229, 0.224, 0.225)
#'
#' # Load an image
#' wmc <- "https://upload.wikimedia.org/wikipedia/commons/thumb/"
#' url <- paste0(wmc, "e/ea/Morsan_Normande_vache.jpg/120px-Morsan_Normande_vache.jpg")
#' img <- base_loader(url)
#'
#' input <- img %>%
#'   transform_to_tensor() %>%
#'   transform_resize(c(800, 800)) %>%
#'   transform_normalize(norm_mean, norm_std)
#' batch <- input$unsqueeze(1)
#'
#' # Mask R-CNN ResNet-50 FPN
#' model <- model_maskrcnn_resnet50_fpn(pretrained = TRUE)
#' model$eval()
#' pred <- model(batch)$detections
#'
#' # Access predictions
#' boxes <- pred$boxes
#' labels <- pred$labels
#' scores <- pred$scores
#' masks <- pred$masks  # Segmentation masks (N, 28, 28)
#'
#' # Visualize boxes
#' if (boxes$size(1) > 0) {
#'   boxed <- draw_bounding_boxes(input, boxes[1:5, ])
#'   tensor_image_browse(boxed)
#' }
#' }
#'
#' @family object_detection_model
#' @name model_maskrcnn
#' @rdname model_maskrcnn
NULL

# Model URLs for pretrained weights
mask_rcnn_model_urls <- list(
  maskrcnn_resnet50 = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/maskrcnn_resnet50.pth",
    "8bbfb4cf0d3fafff09739b15647fd123",
    "170 MB"
  ),
  maskrcnn_resnet50_v2 = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/maskrcnn_resnet50_v2.pth",
    "50aa7c34a52e9a9f16d899db3c56b8e5",
    "178 MB"
  )
)

#' @describeIn model_maskrcnn Mask R-CNN with ResNet-50 FPN
#' @export
model_maskrcnn_resnet50_fpn <- function(pretrained = FALSE, progress = TRUE,
                                        num_classes = 91,
                                        score_thresh = 0.05,
                                        nms_thresh = 0.5,
                                        detections_per_img = 100,
                                        ...) {
  backbone <- resnet_fpn_backbone(pretrained = pretrained)
  model <- maskrcnn_model(backbone, num_classes = num_classes,
                         score_thresh = score_thresh,
                         nms_thresh = nms_thresh,
                         detections_per_img = detections_per_img)()

  if (pretrained && num_classes != 91)
    cli_abort("Pretrained weights require num_classes = 91.")

  if (pretrained) {
    r <- mask_rcnn_model_urls$maskrcnn_resnet50
    name <- "maskrcnn_resnet50_fpn"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "maskrcnn")

    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(.rename_maskrcnn_state_dict(state_dict), strict = FALSE)
  }

  model
}

#' @describeIn model_maskrcnn Mask R-CNN with ResNet-50 FPN V2
#' @export
model_maskrcnn_resnet50_fpn_v2 <- function(pretrained = FALSE, progress = TRUE,
                                           num_classes = 91,
                                           score_thresh = 0.05,
                                           nms_thresh = 0.5,
                                           detections_per_img = 100,
                                           ...) {
  backbone <- resnet_fpn_backbone_v2(pretrained = pretrained)
  model <- maskrcnn_model_v2(backbone, num_classes = num_classes,
                            score_thresh = score_thresh,
                            nms_thresh = nms_thresh,
                            detections_per_img = detections_per_img)()

  if (pretrained && num_classes != 91)
    cli_abort("Pretrained weights require num_classes = 91.")

  if (pretrained) {
    r <- mask_rcnn_model_urls$maskrcnn_resnet50_v2
    name <- "maskrcnn_resnet50_fpn_v2"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "maskrcnn")

    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)

    # Load with flexible matching (similar to fasterrcnn_v2)
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

    model$load_state_dict(state_dict, strict = TRUE)
  }

  model
}

#' @importFrom stats setNames
.rename_maskrcnn_state_dict <- function(state_dict) {
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
