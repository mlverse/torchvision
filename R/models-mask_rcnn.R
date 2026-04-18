#' @include models-faster_rcnn.R
#' @importFrom torch nn_conv_transpose2d torch_int32 torch_float torch_log2 torch_nonzero
NULL

# Mask R-CNN Implementation
# Instance Segmentation model extending Faster R-CNN with mask prediction

# Mask Head Module - Predicts segmentation masks for detected objects
mask_head_module <- torch::nn_module(
    "mask_head",
    initialize = function() {
      # 4 convolutional layers for feature extraction
      self$mask_fcn1 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)
      self$mask_fcn2 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)
      self$mask_fcn3 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)
      self$mask_fcn4 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)

    },
    forward = function(x) {
      x <- nnf_relu(self$mask_fcn1(x))
      x <- nnf_relu(self$mask_fcn2(x))
      x <- nnf_relu(self$mask_fcn3(x))
      nnf_relu(self$mask_fcn4(x))
    }
)

# Mask RCNN predictor - Predicts segmentation masks for detected objects
mask_rcnn_predictor <- torch::nn_module(
    "mask_rcnn_predictor",
    initialize = function(num_classes = 90) {
      # num_classes excludes background, but predictor needs background + all classes
      num_classes_with_bg <- num_classes + 1L

      # Deconvolution layer to upsample from 14x14 to 28x28
      self$conv5_mask <- nn_conv_transpose2d(256, 256, kernel_size = 2, stride = 2)

      # Final 1x1 conv for class-specific mask logits
      self$mask_fcn_logits <- nn_conv2d(256, num_classes_with_bg, kernel_size = 1)
    },
    forward = function(x) {
      x <- nnf_relu(self$conv5_mask(x))
      self$mask_fcn_logits(x)
    }
)

# Mask Head Module V2 - With batch normalization
mask_head_module_v2 <- torch::nn_module(
    "mask_head_v2",
    initialize = function(num_classes = 90) {
      # num_classes excludes background, but predictor needs background + all classes
      num_classes_with_bg <- num_classes + 1L

      # Convolutional blocks with batch normalization
      conv_block <- function() {
        nn_sequential(
          nn_conv2d(256, 256, kernel_size = 3, padding = 1, bias = FALSE),
          nn_batch_norm2d(256),
          nn_relu()
        )
      }

      self$mask_head.0 <- conv_block()
      self$mask_head.1 <- conv_block()
      self$mask_head.2 <- conv_block()
      self$mask_head.3 <- conv_block()

      # Deconvolution without batch norm
      self$mask_predictor.conv5_mask <- nn_sequential(
        nn_conv_transpose2d(256, 256, kernel_size = 2, stride = 2),
        nn_relu()
      )

      # Final 1x1 conv for class-specific mask logits
      self$mask_predictor.mask_fcn_logits <- nn_conv2d(256, num_classes_with_bg, kernel_size = 1)
    },
    forward = function(x) {
      x <- self$mask_head.0(x)
      x <- self$mask_head.1(x)
      x <- self$mask_head.2(x)
      x <- self$mask_head.3(x)
      x <- self$mask_predictor.conv5_mask(x)
      self$mask_predictor.mask_fcn_logits(x)
    }
)


#' ROI Align for Mask Prediction
#'
#' Extracts fixed-size feature maps from regions of interest for mask prediction.
#' Returns a 4D tensor suitable for the mask head (unlike roi_align which flattens).
#'
#' ROI Align for mask prediction – single FPN level
#'
#' This uses grid_sample for bilinear interpolation.
#' It **expects** the proposals in *image* coordinates
#' and applies the supplied `spatial_scale` before the ROI‑Align.
#'
#' @param feature_map   Tensor of shape (1, C, H, W) – one FPN level.
#' @param proposals     Tensor of shape (N, 4) with boxes (x1, y1, x2, y2) in
#'                       the original image coordinate system.
#' @param output_size   integer vector of length 2, e.g. c(14L, 14L).
#' @param spatial_scale Scaling factor between the original image and the
#'                       feature map.  For the first FPN level (stride 4) use
#'                       1/4, for the second level (stride 8) use 1/8, …
#' @param sampling_ratio Integer – number of sampling points per output
#'                       element (default = 2, the same as Torchvision).
#' @param aligned       Logical – if TRUE uses the “aligned” mode (default FALSE).
#'
#' @return Tensor of shape (N, C, output_size\[1\], output_size\[2\]).
#' @noRd
roi_align_masks <- function(feature_map,
                            proposals,
                            output_size = c(14L, 14L),
                            spatial_scale = 1.0,
                            sampling_ratio = 2L,
                            aligned = FALSE) {

  # Scale the boxes to the feature map resolution
  boxes_scaled <- proposals$to(dtype = torch_float())$mul(spatial_scale)

  num_rois <- boxes_scaled$size(1)
  if (num_rois == 0) {
    return(torch_empty(c(0, feature_map$size(2), output_size[1], output_size[2]),
                       device = feature_map$device))
  }

  channels <- feature_map$size(2)
  h_feat <- feature_map$size(3)
  w_feat <- feature_map$size(4)

  # Normalize coordinates to match grid_sample [-1 to 1]
  x1 <- (boxes_scaled[, 1] / (w_feat - 1) * 2) - 1
  y1 <- (boxes_scaled[, 2] / (h_feat - 1) * 2) - 1
  x2 <- (boxes_scaled[, 3] / (w_feat - 1) * 2) - 1
  y2 <- (boxes_scaled[, 4] / (h_feat - 1) * 2) - 1

  # Create a grid of output_size
  grid_y <- torch_linspace(0, 1, output_size[1], device = feature_map$device)
  grid_x <- torch_linspace(0, 1, output_size[2], device = feature_map$device)

  # Meshgrid to get relative coordinates
  grids <- torch_meshgrid(list(grid_y, grid_x), indexing = "ij")
  rel_y <- grids[[1]]
  rel_x <- grids[[2]]

  # Linear interpolation for each ROI [N, output_size[1], output_size[2]]
  sampling_x <- x1$view(c(-1, 1, 1)) + rel_x$view(c(1, output_size[1], output_size[2])) * (x2 - x1)$view(c(-1, 1, 1))
  sampling_y <- y1$view(c(-1, 1, 1)) + rel_y$view(c(1, output_size[1], output_size[2])) * (y2 - y1)$view(c(-1, 1, 1))

  # Clamp grid to [-1, 1] to avoid needing padding (especially for MPS compatibility)
  sampling_x <- sampling_x$clamp(-1, 1)
  sampling_y <- sampling_y$clamp(-1, 1)

  # Concat to get a grid of [N, output_size[1], output_size[2], 2]
  grid <- torch_stack(list(sampling_x, sampling_y), dim = -1)

  # Bilinear sampling
  # Feature map is (1, C, H, W), expand to (N, C, H, W) for all ROIs
  input_expanded <- feature_map$squeeze(1)$unsqueeze(1)$expand(c(num_rois, channels, h_feat, w_feat))

  pooled_features <- nnf_grid_sample(
    input_expanded,
    grid,
    mode = "bilinear",
    padding_mode = "zeros",
    align_corners = FALSE
  )

  # Return [N, C, output_size[1], output_size[2]]
  pooled_features
}


#' Multi‑scale ROI Align for Mask‑RCNN (FPN)
#'
#'
#' @param feature_maps   List of 4 tensors, each of shape (1, C, H_l, W_l)
#'                       – the four FPN levels P2, P3, P4, P5.
#' @param proposals      Tensor (N, 4) – boxes in image coordinates.
#' @param output_size    Integer vector, default c(14L, 14L).
#' @param sampling_ratio Sampling ratio passed to the low‑level op (default 2L).
#' @param aligned        Whether to use the aligned ROI‑Align mode (default FALSE).
#'
#' @return Tensor (N, C, out_h, out_w) – pooled mask features.
#' @noRd
roi_align_masks_fpn <- function(feature_maps,
                                proposals,
                                output_size = c(14L, 14L),
                                sampling_ratio = 2L,
                                aligned = FALSE) {
  #  Compute the level for each ROI (FPN paper Eq. (1))
  # width and height in original image space
  widths  <- proposals[, 3] - proposals[, 1]
  heights <- proposals[, 4] - proposals[, 2]

  # Guard against degenerate boxes (avoid log2(0))
  widths[widths <= 0]   <- 1
  heights[heights <= 0] <- 1

  # sqrt(area) / 224  (224 is the reference size in the paper)
  area_sqrt <- (widths * heights)$sqrt()
  target_level <- (torch_log2(area_sqrt / 224) + 4)$floor()   # level in {2,3,4,5}
  target_level <- torch_clamp(target_level, min = 2, max = 5) # clamp to existing levels

  # Convert from the level {2,3,4,5} to an index in feature_maps list (1‑based)
  level_idx <- (target_level - 2L)$to(dtype = torch_long()) + 1L   # 1,2,3,4

  #  Allocate output tensor
  n_boxes   <- proposals$size(1)
  c_channels <- feature_maps[[1]]$size(2)    # same number of channels for all levels
  out       <- torch_zeros(c(n_boxes, c_channels,
                             output_size[1], output_size[2]),
                           dtype = feature_maps[[1]]$dtype,
                           device = feature_maps[[1]]$device)

  #  Loop over the four levels and pool the ROIs belonging to it
  spatial_scales <- list(1/4, 1/8, 1/16, 1/32)   # stride of each level

  for (lvl in seq_along(feature_maps)) {
    # mask of boxes that belong to the current level
    lvl_mask <- level_idx$eq(lvl)$squeeze()
    idx      <- torch_nonzero(lvl_mask)$squeeze()   # indices

    # Check if there are any boxes at this level
    if (idx$numel() == 0) next

    # Ensure idx is 1D for consistent indexing
    if (idx$dim() == 0) idx <- idx$unsqueeze(1)  # scalar -> (1,)

    # pick the subset of proposals that belong to this level
    rois_lvl <- proposals[idx, , drop = FALSE]  # idx is already 1-based

    # call the *single‑scale* wrapper (it adds the batch column internally)
    pooled_lvl <- roi_align_masks(
      feature_map   = feature_maps[[lvl]],
      proposals     = rois_lvl,
      output_size   = output_size,
      spatial_scale = spatial_scales[[lvl]],
      sampling_ratio = sampling_ratio,
      aligned        = aligned
    )   # shape (M, C, out_h, out_w)

    # write the result back into the pre‑allocated output tensor
    out[idx, , , ] <- pooled_lvl  # idx is already 1-based
  }

  out
}

# Mask R-CNN Model - Extends Faster R-CNN with mask prediction
maskrcnn_model <- torch::nn_module(
  "maskrcnn_model",
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

      # RPN (Region Proposal Network)
      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head(in_channels = backbone$out_channels)
        },
        forward = function(features) {
          self$head(features)
        }
      )()

      # ROI heads for box prediction
      self$roi_heads <- roi_heads_module(num_classes = num_classes)

      # Mask head without mask predictor
      self$mask_head <- mask_head_module()

      # Mask predictor for mask prediction
      self$mask_predictor <- mask_rcnn_predictor(num_classes = num_classes)
    },

    forward = function(images) {
      features <- self$backbone(images)
      rpn_out <- self$rpn(features)

      batch_size <- images$shape[1]
      image_size <- images$shape[3:4]
      final_results <- list()

      for (b in seq_len(batch_size)) {

        props <- generate_proposals(features, rpn_out, image_size, c(4, 8, 16, 32),
                                    batch_idx = b, score_thresh = 0,
                                    nms_thresh = self$nms_thresh)

        if (props$proposals$shape[1] == 0) {
          empty <- list(
            boxes = torch::torch_empty(c(0, 4)),
            labels = torch::torch_empty(c(0), dtype = torch::torch_long()),
            scores = torch::torch_empty(c(0)),
            masks = torch::torch_empty(c(0, 28, 28))
          )
          final_results[[b]] <- empty
          next
        }

        # Limit proposals to avoid slow ROI pooling
        max_proposals <- 1000
        if (props$proposals$shape[1] > max_proposals) {
          props$proposals <- props$proposals[1:max_proposals, ]
        }

        # Get ROI head predictions
        roi_out <- self$roi_heads(features, props$proposals, batch_idx = b)

        # Postprocess detections
        det_result <- postprocess_detections(
          class_logits = roi_out$scores,
          box_regression = roi_out$boxes,
          proposals = props$proposals,
          image_size = image_size,
          num_classes = self$num_classes,
          score_thresh = self$score_thresh,
          nms_thresh = self$nms_thresh,
          detections_per_img = self$detections_per_img
        )

        final_boxes <- det_result$boxes
        final_labels <- det_result$labels
        final_scores <- det_result$scores

        # Predict masks for detected objects
        if (final_boxes$shape[1] > 0) {
          # For mask prediction, we need to map final boxes back to proposals
          # Since postprocess_detections decodes boxes, we need the original proposals
          # that correspond to the kept detections. We'll use the final_boxes directly.

          mask_features <- roi_align_masks_fpn(
            feature_maps = features,
            proposals    = final_boxes,
            output_size  = c(14L, 14L),
            sampling_ratio = 2L,
            aligned        = FALSE
          )
          mask_conv <- self$mask_head(mask_features)
          mask_logits <- self$mask_predictor(mask_conv)  # Shape: (N, num_classes, 28, 28)

          # Extract masks for predicted classes
          n_kept <- final_labels$shape[1]
          final_masks <- torch::torch_zeros(c(n_kept, 28, 28), device = final_boxes$device)

          for (i in seq_len(n_kept)) {
            class_idx <- as.integer(final_labels[i]$item())
            # mask_logits has channels [background, class1, ..., class90]
            # labels are [1, 2, ..., 90], so we need to skip channel 1 (background)
            # In 1-based R indexing: label 1 should access channel 2 (class 1)
            final_masks[i, , ] <- mask_logits[i, class_idx + 1L, , ]
          }

          # Apply sigmoid to get probabilities
          final_masks <- torch::torch_sigmoid(final_masks)

        } else {
          final_masks <- torch::torch_empty(c(0, 28, 28))
        }

        final_results[[b]] <- list(
          boxes = final_boxes,
          labels = final_labels,
          scores = final_scores,
          masks = final_masks
        )
      }
      list(features = features, detections = final_results)
    }
)


# Mask R-CNN Model V2 - With batch normalization
maskrcnn_model_v2 <- torch::nn_module(
  "maskrcnn_model_v2",
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

      # RPN with V2 head
      self$rpn <- torch::nn_module(
        initialize = function() {
          self$head <- rpn_head_v2(in_channels = backbone$out_channels)
        },
        forward = function(features) {
          self$head(features)
        }
      )()

      # ROI heads V2 for box prediction
      self$roi_heads <- roi_heads_module_v2(num_classes = num_classes)

      # Mask head V2 for mask prediction
      self$mask_head <- mask_head_module_v2(num_classes = num_classes)

    },

    forward = function(images) {
      features <- self$backbone(images)
      rpn_out <- self$rpn(features)

      batch_size <- images$shape[1]
      image_size <- images$shape[3:4]
      final_results <- list()

      for (b in seq_len(batch_size)) {
        props <- generate_proposals(features, rpn_out, image_size, c(4, 8, 16, 32),
                                    batch_idx = b, score_thresh = 0,
                                    nms_thresh = self$nms_thresh)

        if (props$proposals$shape[1] == 0L) {
          empty <- list(
            boxes = torch::torch_empty(c(0, 4)),
            labels = torch::torch_empty(c(0), dtype = torch::torch_long()),
            scores = torch::torch_empty(c(0)),
            masks = torch::torch_empty(c(0, 28, 28))
          )
          final_results[[b]] <- empty
          next
        }

        # Limit proposals to avoid slow ROI pooling
        max_proposals <- 1000
        if (props$proposals$shape[1] > max_proposals) {
          props$proposals <- props$proposals[1:max_proposals, ]
        }

        # Get ROI head predictions
        roi_out <- self$roi_heads(features, props$proposals, batch_idx = b)

        # Postprocess detections
        det_result <- postprocess_detections(
          class_logits = roi_out$scores,
          box_regression = roi_out$boxes,
          proposals = props$proposals,
          image_size = image_size,
          num_classes = self$num_classes,
          score_thresh = self$score_thresh,
          nms_thresh = self$nms_thresh,
          detections_per_img = self$detections_per_img
        )

        final_boxes <- det_result$boxes
        final_labels <- det_result$labels
        final_scores <- det_result$scores

        # Predict masks for detected objects
        if (final_boxes$shape[1] > 0) {
          mask_features <- roi_align_masks_fpn(
            feature_maps = features,
            proposals    = final_boxes,
            output_size  = c(14L, 14L),
            sampling_ratio = 2L,
            aligned        = FALSE
          )
          mask_logits <- self$mask_head(mask_features)  # (N, num_classes, 28, 28)

          # Extract masks for predicted classes
          n_kept <- final_labels$shape[1]
          final_masks <- torch::torch_zeros(c(n_kept, 28, 28), device = final_boxes$device)

          for (i in seq_len(n_kept)) {
            class_idx <- as.integer(final_labels[i]$item())
            # mask_logits has channels [background, class1, ..., class90]
            # labels are [1, 2, ..., 90], so we need to skip channel 1 (background)
            # In 1-based R indexing: label 1 should access channel 2 (class 1)
            final_masks[i, , ] <- mask_logits[i, class_idx + 1L, , ]
          }

          # Apply sigmoid to get probabilities
          final_masks <- torch::torch_sigmoid(final_masks)

        } else {
          final_masks <- torch::torch_empty(c(0, 28, 28))
        }

        final_results[[b]] <- list(
          boxes = final_boxes,
          labels = final_labels,
          scores = final_scores,
          masks = final_masks
        )
    }
  list(features = features, detections = final_results)
  }
)


#' Mask R-CNN Models
#'
#' Construct Mask R-CNN model variants for instance segmentation task.
#' Mask R-CNN extends Faster R-CNN by adding a mask prediction branch that
#' outputs segmentation masks for each detected object.
#'
#' @param pretrained Logical. If TRUE, loads pretrained weights from local file.
#' @param progress Logical. Show progress bar during download (unused).
#' @param num_classes Number of output classes excluding background (default: 90 for COCO).
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
#' # ImageNet normalization constants, see https://pytorch.org/vision/stable/models.html
#' norm_mean <- c(0.485, 0.456, 0.406)
#' norm_std  <- c(0.229, 0.224, 0.225)
#'
#' # Load an image
#' url <- paste0("https://upload.wikimedia.org/wikipedia/commons/thumb/",
#'               "e/ea/Morsan_Normande_vache.jpg/120px-Morsan_Normande_vache.jpg")
#' img <- base_loader(url)
#'
#' input <- img %>%
#'   transform_to_tensor() %>%
#'   transform_resize(c(800, 800)) %>%
#'   transform_normalize(norm_mean, norm_std)
#' batch <- input$unsqueeze(1)
#'
#' # Mask R-CNN ResNet-50 FPN
#' model <- model_maskrcnn_resnet50_fpn(pretrained = TRUE, , detections_per_img = 5)
#' model$eval()
#'
#' torch::with_no_grad({pred <- model(batch)$detections[[1]]})
#'
#' # Visualize boxes
#' labels <- coco_classes(as.integer(pred$labels))
#' boxed <- draw_bounding_boxes(image, pred$boxes, labels = labels)
#' tensor_image_browse(boxed)
#'}
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
                                        num_classes = 90,
                                        score_thresh = 0.05,
                                        nms_thresh = 0.5,
                                        detections_per_img = 100,
                                        ...) {
  backbone <- resnet_fpn_backbone(pretrained = pretrained)
  model <- maskrcnn_model(backbone, num_classes = num_classes,
                         score_thresh = score_thresh,
                         nms_thresh = nms_thresh,
                         detections_per_img = detections_per_img)

  if (pretrained && num_classes != 90)
    cli_abort("Pretrained weights require num_classes = 90 (excluding background).")

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
                                           num_classes = 90,
                                           score_thresh = 0.05,
                                           nms_thresh = 0.5,
                                           detections_per_img = 100,
                                           ...) {
  backbone <- resnet_fpn_backbone_v2(pretrained = pretrained)
  model <- maskrcnn_model_v2(backbone, num_classes = num_classes,
                            score_thresh = score_thresh,
                            nms_thresh = nms_thresh,
                            detections_per_img = detections_per_img)

  if (pretrained && num_classes != 90)
    cli_abort("Pretrained weights require num_classes = 90 (excluding background).")

  if (pretrained) {
    r <- mask_rcnn_model_urls$maskrcnn_resnet50_v2
    name <- "maskrcnn_resnet50_fpn_v2"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "maskrcnn")

    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)

    # Rename state dict keys to match model structure
    state_dict <- .rename_maskrcnn_state_dict_v2(state_dict)

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
    sub(pattern = "(rpn\\.head\\.conv\\.)", replacement = "\\10\\.0\\.", x = .) %>%
    # remove roi_head prefix to mask_head
    sub(pattern = "roi_heads\\.mask", replacement = "mask", x = .)

  # Recreate a list with renamed keys
  setNames(state_dict[names(state_dict)], new_names)
}

.rename_maskrcnn_state_dict_v2 <- function(state_dict) {
  . <- NULL # Nulling strategy for no visible binding check Note
  new_names <- names(state_dict) %>%
    # change roi_head prefix into mask_head.mask_
    sub(pattern = "roi_heads\\.mask_", replacement = "mask_head\\.mask_", x = .) %>%
    sub(pattern = "roi_heads\\.box_head\\.5\\.", replacement = "roi_heads\\.box_head\\.4\\.", x = .) %>%
    sub(pattern = "(mask_head\\.mask_predictor\\.conv5_mask)", replacement = "\\1\\.0", x = .)

  # Recreate a list with renamed keys
  setNames(state_dict[names(state_dict)], new_names)
}
