# DeepLabV3 model family with auxiliary classifier

#' DeepLabV3 Models
#'
#' Constructs DeepLabV3 semantic segmentation models with a ResNet backbone as
#' described in \emph{Rethinking Atrous Convolution for Semantic Image
#' Segmentation}. These models employ atrous spatial pyramid pooling to capture
#' multi-scale context.
#'
#' @section Task:
#' Semantic image segmentation with 21 output classes by default (COCO).
#'
#' @section Input Format:
#' The models expect input tensors of shape \code{(batch_size, 3, H, W)}. Typical
#' training uses 520x520 images.
#'
#' @inheritParams model_resnet18
#' @param num_classes Number of output classes.
#' @param aux_loss Logical or NULL. If `TRUE`, includes an auxiliary classifier branch.
#'   If `NULL` (default), the presence of aux classifier is inferred from pretrained weights.
#' @param pretrained_backbone If `TRUE` and `pretrained = FALSE`, loads
#'   ImageNet weights for the ResNet backbone.
#' @param ... Other parameters passed to the model implementation.
#'
#' @importFrom torch nn_module
#' @family models
#'
#' @examples
#' \dontrun{
#' # VOC class names (used by default 21-class COCO/Pascal VOC models)
#' voc_classes <- c(
#'   "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
#'   "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse",
#'   "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
#' )
#'
#' model <- model_deeplabv3_resnet50(pretrained = TRUE)
#' model$eval()
#' image_batch <- torch::torch_randn(1, 3, 520, 520)
#' output <- model(image_batch)
#' predicted_class <- output$out[1, , 260, 260]$argmax()$item() + 1
#' class_label <- voc_classes[predicted_class]
#' print(paste("Predicted class at (260, 260):", class_label))
#'
#'
#' model <- model_deeplabv3_resnet101(
#'   pretrained = FALSE,
#'   num_classes = 3,
#'   aux_loss = TRUE
#' )
#' image_batch <- torch::torch_randn(1, 3, 520, 520)
#' output <- model(image_batch)
#' # Check output shapes
#' dim(output$out)  # e.g., (1, 3, 520, 520)
#' dim(output$aux)  # e.g., (1, 3, 520, 520)
#' }
#' @name model_deeplabv3
#' @rdname model_deeplabv3
NULL

deeplabv3_model_urls <- c(
  "deeplabv3_resnet50_coco" = "https://torch-cdn.mlverse.org/models/vision/v1/models/deeplabv3_resnet50_coco.pth",
  "deeplabv3_resnet101_coco" = "https://torch-cdn.mlverse.org/models/vision/v1/models/deeplabv3_resnet101_coco.pth"
)

# ASPP module matching PyTorch
aspp_module <- nn_module(
  "ASPP",
  initialize = function(in_channels, out_channels, atrous_rates) {
    self$convs <- nn_module_list()

    self$convs$append(nn_sequential(
      nn_conv2d(in_channels, out_channels, kernel_size = 1, bias = FALSE),
      nn_batch_norm2d(out_channels),
      nn_relu()
    ))

    for (rate in atrous_rates) {
      self$convs$append(nn_sequential(
        nn_conv2d(in_channels, out_channels, kernel_size = 3, padding = rate, dilation = rate, bias = FALSE),
        nn_batch_norm2d(out_channels),
        nn_relu()
      ))
    }

    self$convs$append(nn_sequential(
      nn_adaptive_avg_pool2d(output_size = c(1, 1)),
      nn_conv2d(in_channels, out_channels, kernel_size = 1, bias = FALSE),
      nn_batch_norm2d(out_channels),
      nn_relu()
    ))

    self$project <- nn_sequential(
      nn_conv2d((length(atrous_rates) + 2) * out_channels, out_channels, kernel_size = 1, bias = FALSE),
      nn_batch_norm2d(out_channels),
      nn_relu(),
      nn_dropout(0.5)
    )
  },
  forward = function(x) {
    input_size <- x$shape[3:4]  # Get height and width dimensions
    res <- list()

    for (i in 1:(length(self$convs) - 1)) {
      res[[i]] <- self$convs[[i]](x)
    }

    global_feat <- self$convs[[length(self$convs)]](x)
    target_size <- as.integer(input_size)
    global_feat <- nnf_interpolate(global_feat, size = target_size, mode = "bilinear", align_corners = FALSE)
    res[[length(res) + 1]] <- global_feat

    x <- torch_cat(res, dim = 2)
    self$project(x)
  }
)

# Main classifier head
deeplab_head <- function(in_channels, num_classes) {
  nn_sequential(
    aspp_module(in_channels, 256, atrous_rates = c(12, 24, 36)),
    nn_conv2d(256, 256, kernel_size = 3, padding = 1, bias = FALSE),
    nn_batch_norm2d(256),
    nn_relu(),
    nn_conv2d(256, num_classes, kernel_size = 1)
  )
}

# Auxiliary classifier head (simpler, operates on layer3 features)
aux_classifier_head <- function(in_channels, num_classes) {
  nn_sequential(
    nn_conv2d(in_channels, 256, kernel_size = 3, padding = 1, bias = FALSE),
    nn_batch_norm2d(256),
    nn_relu(),
    nn_dropout(0.1),
    nn_conv2d(256, num_classes, kernel_size = 1)
  )
}

# Updated DeepLabV3 wrapper with auxiliary classifier
DeepLabV3 <- nn_module(
  "DeepLabV3",
  initialize = function(backbone, classifier, aux_classifier = NULL) {
    self$backbone <- backbone
    self$classifier <- classifier
    self$aux_classifier <- aux_classifier
  },
  forward = function(x) {
    input_shape <- x$shape[3:4]  # Get height and width dimensions

    x <- self$backbone$conv1(x)
    x <- self$backbone$bn1(x)
    x <- self$backbone$relu(x)
    x <- self$backbone$maxpool(x)
    x <- self$backbone$layer1(x)
    x <- self$backbone$layer2(x)
    x <- self$backbone$layer3(x)

    aux_x <- x

    x <- self$backbone$layer4(x)

    # Main classifier
    x <- self$classifier(x)
    target_size <- as.integer(input_shape)
    x <- nnf_interpolate(x, size = target_size, mode = "bilinear", align_corners = FALSE)

    result <- list(out = x)

    # Auxiliary classifier (if present)
    if (!is.null(self$aux_classifier)) {
      aux_out <- self$aux_classifier(aux_x)
      aux_out <- nnf_interpolate(aux_out, size = target_size, mode = "bilinear", align_corners = FALSE)
      result$aux <- aux_out
    }

    result
  }
)

# Model constructor
deeplabv3_resnet_factory <- function(arch, block, layers, pretrained, progress,
                                     num_classes, aux_loss = NULL,
                                     pretrained_backbone = FALSE, ...) {
  if (pretrained && num_classes != 21) {
    cli_abort("Pretrained weights require num_classes = 21.")
  }

  if (is.null(aux_loss))
    aux_loss <- FALSE

  if (pretrained && pretrained_backbone)
    cli_warn("`pretrained_backbone` ignored when `pretrained = TRUE`." )

  backbone_arch <- sub("deeplabv3_", "", arch)

  if (pretrained_backbone && !pretrained) {
    backbone <- .resnet(backbone_arch, block, layers, pretrained = TRUE,
                        progress = progress,
                        replace_stride_with_dilation = c(FALSE, TRUE, TRUE), ...)
  } else {
    backbone <- resnet(block, layers,
                       replace_stride_with_dilation = c(FALSE, TRUE, TRUE), ...)
  }
  backbone$fc <- nn_identity()
  backbone$avgpool <- nn_identity()

  classifier <- deeplab_head(2048, num_classes)

  aux_classifier <- NULL
  if (aux_loss) {
    aux_classifier <- aux_classifier_head(1024, num_classes)
  }

  model <- DeepLabV3(backbone, classifier, aux_classifier)

  if (pretrained) {
    state_dict_path <- download_and_cache(deeplabv3_model_urls[[paste0(arch, "_coco")]])
    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }

  model
}

#' @describeIn model_deeplabv3 DeepLabV3 with ResNet-50 backbone
#' @export
model_deeplabv3_resnet50 <- function(pretrained = FALSE, progress = TRUE,
                                     num_classes = 21, aux_loss = NULL,
                                     pretrained_backbone = FALSE, ...) {
  deeplabv3_resnet_factory(
    "deeplabv3_resnet50", bottleneck, c(3, 4, 6, 3),
    pretrained, progress, num_classes, aux_loss, pretrained_backbone, ...
  )
}

#' @describeIn model_deeplabv3 DeepLabV3 with ResNet-101 backbone
#' @export
model_deeplabv3_resnet101 <- function(pretrained = FALSE, progress = TRUE,
                                      num_classes = 21, aux_loss = NULL,
                                      pretrained_backbone = FALSE, ...) {
  deeplabv3_resnet_factory(
    "deeplabv3_resnet101", bottleneck, c(3, 4, 23, 3),
    pretrained, progress, num_classes, aux_loss, pretrained_backbone, ...
  )
}
