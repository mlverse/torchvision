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


# Mask R-CNN Model - Extends Faster R-CNN with mask prediction
maskrcnn_model <- torch::nn_module(
  "maskrcnn_model",
    initialize = function(backbone, num_classes,
                          score_thresh = 0.05,
                          nms_thresh = 0.5,
                          detections_per_img = 100,
                          rpn_config = NULL) {
      # resolved here: nn_module() evaluates default arguments outside the package namespace
      self$rpn_config <- if (is.null(rpn_config)) rcnn_resnet_rpn_config() else rpn_config
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
      rcnn_forward(self, images)
    }
)


# Mask R-CNN Model V2 - With batch normalization
maskrcnn_model_v2 <- torch::nn_module(
  "maskrcnn_model_v2",
    initialize = function(backbone, num_classes,
                          score_thresh = 0.05,
                          nms_thresh = 0.5,
                          detections_per_img = 100,
                          rpn_config = NULL) {
      # resolved here: nn_module() evaluates default arguments outside the package namespace
      self$rpn_config <- if (is.null(rpn_config)) rcnn_resnet_rpn_config() else rpn_config
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
      rcnn_forward(self, images)
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
#'       \item `masks`: Mask probabilities pasted into the image, (N, H, W)
#'       \item `mask_probs`: Mask probabilities in the box frame, (N, 28, 28)
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
#' model <- model_maskrcnn_resnet50_fpn(pretrained = TRUE, detections_per_img = 5)
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
  backbone <- resnet_fpn_backbone(pretrained = pretrained, progress = progress)
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
    state_dict_path <- download_and_cache(r[1], prefix = "maskrcnn", progress = progress)

    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)
    .load_rcnn_state_dict(model, .rename_maskrcnn_state_dict(state_dict))
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
  backbone <- resnet_fpn_backbone_v2(pretrained = pretrained, progress = progress)
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
    state_dict_path <- download_and_cache(r[1], prefix = "maskrcnn", progress = progress)

    if (!tools::md5sum(state_dict_path) == r[2]) {
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }

    state_dict <- torch::load_state_dict(state_dict_path)

    # Rename state dict keys to match model structure
    state_dict <- .rename_maskrcnn_state_dict_v2(state_dict)

    .load_rcnn_state_dict(model, state_dict)
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
    sub(pattern = "(mask_head\\.mask_predictor\\.conv5_mask)", replacement = "\\1\\.0", x = .)

  # Recreate a list with renamed keys
  setNames(state_dict[names(state_dict)], new_names)
}
